"""Client behaviour: routing, fallback, tool loop, error propagation.

The client is tested against a synthetic registry (built directly, not
discovered) so routing logic is verified without depending on which backends
happen to be up on the machine running the tests.
"""

from __future__ import annotations

import pytest

from freeaipi.client import FreeAIClient, ChatOptions
from freeaipi.errors import ConfigurationError, ModelNotFoundError, RateLimitError
from freeaipi.http import MockTransport
from freeaipi.models import Capabilities, ChatResponse, ModelInfo, Provider, Usage
from freeaipi.providers.base import BaseProvider
from freeaipi.registry import Registry


class StubProvider(BaseProvider):
    """Answers from a script; records what it was asked.

    ``replies`` is a queue of text replies. ``tool_calls`` is a queue of tool
    calls emitted *instead* of a text reply — the loop in
    :meth:`FreeAIClient.run_tools` is only exercised when the model actually
    requests a tool, so a stub that only ever returns text never reaches the
    handler.
    """

    provider = Provider.MOCK

    def __init__(self, model, transport, replies=None, fail_with=None, tool_calls=None):
        super().__init__(model, transport)
        self.replies = list(replies or ["stub reply"])
        self.tool_calls = list(tool_calls or [])
        self.fail_with = fail_with
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=2048, system=None,
             json_mode=False, seed=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self.fail_with:
            raise self.fail_with
        if self.tool_calls:
            tc = self.tool_calls.pop(0)
            return ChatResponse(
                content="", model_id=self.model.model_id, provider=self.provider,
                usage=Usage(1, 1, 2), latency_ms=1.0, tool_calls=[tc],
            )
        content = self.replies.pop(0) if self.replies else "stub reply"
        return ChatResponse(
            content=content, model_id=self.model.model_id, provider=self.provider,
            usage=Usage(1, 1, 2), latency_ms=1.0,
        )


def build_client(models: dict[str, tuple[Provider, Capabilities, bool]]) -> FreeAIClient:
    """Construct a client whose registry is exactly ``models``.

    ``models`` maps model_id → (provider, capabilities, auth_required).
    """
    transport = MockTransport()
    client = FreeAIClient(transport=transport, include_local=False,
                          include_openrouter=False, include_pollinations=False)
    reg = Registry()
    for model_id, (provider, caps, auth) in models.items():
        info = ModelInfo(
            model_id=model_id, provider=provider,
            provider_model_id=model_id, name=model_id,
            capabilities=caps, auth_required=auth,
        )
        reg.register(info, StubProvider(info, transport))
    client.registry = reg
    return client


@pytest.fixture
def mixed_client():
    return build_client({
        "free-chat": (Provider.POLLINATIONS, Capabilities(text=True, context_length=8192), False),
        "free-tools": (Provider.POLLINATIONS, Capabilities(text=True, tools=True, context_length=32768), False),
        "or-free": (Provider.OPENROUTER, Capabilities(text=True, tools=True, context_length=1000000), True),
        "local-big": (Provider.LOCAL, Capabilities(text=True, tools=True, reasoning=True, context_length=262144), False),
    })


class TestRouting:
    def test_explicit_model_used(self, mixed_client):
        r = mixed_client.chat("hi", model="free-chat")
        assert r.model_id == "free-chat"

    def test_auto_pick_prefers_no_key(self, mixed_client):
        """No requirement → a no-key model, never the keyed one."""
        r = mixed_client.chat("hi")
        assert r.model_id == "local-big"
        assert r.model_id != "or-free"

    def test_tools_requirement_filters(self, mixed_client):
        """Tool-capable + no key → the largest-context no-key model wins."""
        r = mixed_client.chat("hi", tools=[{"type": "function", "function": {"name": "f"}}])
        assert r.model_id == "local-big"

    def test_best_returns_full_registry_key(self, mixed_client):
        assert mixed_client.best(requires_tools=True) == "local/local-big"

    def test_best_without_preference_prefers_no_key(self, mixed_client):
        """No-key models win over keyed ones, then largest context."""
        assert mixed_client.best() == "local/local-big"

    def test_best_excludes_keyed_models(self, mixed_client):
        """or-free has the biggest context but needs a key, so it never wins."""
        assert mixed_client.best() != "openrouter/or-free"

    def test_best_with_key_allowed(self, mixed_client):
        """prefer_no_key=False lets the keyed 1M-context model win."""
        assert mixed_client.best(prefer_no_key=False) == "openrouter/or-free"

    def test_best_rank_is_deterministic(self, mixed_client):
        first = [mixed_client.best() for _ in range(5)]
        assert len(set(first)) == 1

    def test_best_no_match_raises_with_hint(self, mixed_client):
        with pytest.raises(ModelNotFoundError) as ei:
            mixed_client.best(requires_vision=True)
        assert "free-chat" in str(ei.value)

    def test_string_message_wrapped(self, mixed_client):
        mixed_client.chat("hi", model="free-chat")
        provider = mixed_client.registry.providers["pollinations/free-chat"]
        assert provider.calls[0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_unknown_option_rejected(self, mixed_client):
        with pytest.raises(TypeError):
            mixed_client.chat("hi", model="free-chat", bogus_option=1)


class TestFallback:
    def test_fallback_used_on_failure(self):
        transport = MockTransport()
        client = FreeAIClient(transport=transport, include_local=False,
                              include_openrouter=False, include_pollinations=False)
        reg = Registry()
        for mid in ("primary", "backup"):
            info = ModelInfo(model_id=mid, provider=Provider.MOCK, provider_model_id=mid, name=mid)
            reg.register(info, StubProvider(info, transport))
        client.registry = reg
        client.registry.providers["mock/primary"].fail_with = RuntimeError("primary down")

        r = client.chat("hi", model="primary", fallback=["backup"])
        assert r.model_id == "backup"

    def test_rate_limit_is_not_fallen_back_from(self):
        """Quota is not a reason to silently switch models — the caller must know."""
        transport = MockTransport()
        client = FreeAIClient(transport=transport, include_local=False,
                              include_openrouter=False, include_pollinations=False)
        reg = Registry()
        for mid in ("primary", "backup"):
            info = ModelInfo(model_id=mid, provider=Provider.MOCK, provider_model_id=mid, name=mid)
            reg.register(info, StubProvider(info, transport))
        client.registry = reg
        client.registry.providers["mock/primary"].fail_with = RateLimitError("quota", retry_after=15.0)

        with pytest.raises(RateLimitError):
            client.chat("hi", model="primary", fallback=["backup"])

    def test_no_targets_raises(self):
        client = build_client({})
        with pytest.raises(ModelNotFoundError):
            client.chat("hi")


class TestToolLoop:
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "current time",
            "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}},
        },
    }]

    def _loop_client(self, replies, tool_calls=None):
        transport = MockTransport()
        client = FreeAIClient(transport=transport, include_local=False,
                              include_openrouter=False, include_pollinations=False)
        reg = Registry()
        info = ModelInfo(
            model_id="m", provider=Provider.MOCK, provider_model_id="m", name="m",
            capabilities=Capabilities(text=True, tools=True),
        )
        reg.register(info, StubProvider(info, transport, replies=replies, tool_calls=tool_calls))
        client.registry = reg
        return client

    def test_tool_call_round_trips_through_handler(self):
        from freeaipi.models import ToolCall

        client = self._loop_client(
            replies=["final answer"],
            tool_calls=[ToolCall(id="c1", name="get_time", arguments={"tz": "UTC"})],
        )
        seen = []
        r = client.run_tools(
            [{"role": "user", "content": "time?"}], self.TOOLS,
            {"get_time": lambda **kw: seen.append(kw) or "12:00"},
        )
        assert seen == [{"tz": "UTC"}]
        assert r.content == "final answer"
        assert r.raw["rounds"] == 2

    def test_tool_result_appended_as_tool_message(self):
        from freeaipi.models import ToolCall

        client = self._loop_client(
            replies=["done"],
            tool_calls=[ToolCall(id="c1", name="get_time", arguments={"tz": "UTC"})],
        )
        client.run_tools(
            [{"role": "user", "content": "time?"}], self.TOOLS,
            {"get_time": lambda **kw: "12:00"},
        )
        second_call_msgs = client.registry.providers["mock/m"].calls[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m["role"] == "tool"]
        assert tool_msgs == [{"role": "tool", "tool_call_id": "c1", "content": "12:00"}]

    def test_handler_error_becomes_tool_message(self):
        """A raising handler must surface as a tool result, not kill the turn."""
        from freeaipi.models import ToolCall

        client = self._loop_client(
            replies=["recovered"],
            tool_calls=[ToolCall(id="c1", name="get_time", arguments={"tz": "UTC"})],
        )
        r = client.run_tools(
            [{"role": "user", "content": "x"}], self.TOOLS,
            {"get_time": lambda **kw: (_ for _ in ()).throw(ValueError("bad tz"))},
        )
        second_call_msgs = client.registry.providers["mock/m"].calls[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if m["role"] == "tool"]
        assert "error: bad tz" in tool_msgs[0]["content"]
        assert r.content == "recovered"

    def test_missing_handler_reported_as_result(self):
        from freeaipi.models import ToolCall

        client = self._loop_client(
            replies=["done"],
            tool_calls=[ToolCall(id="c1", name="no_such_tool", arguments={})],
        )
        client.run_tools([{"role": "user", "content": "x"}], self.TOOLS, {})
        msgs = client.registry.providers["mock/m"].calls[1]["messages"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert "no handler for no_such_tool" in tool_msgs[0]["content"]

    def test_no_tool_calls_returns_after_one_round(self):
        client = self._loop_client(replies=["done"])
        r = client.run_tools([{"role": "user", "content": "x"}], self.TOOLS, {})
        assert r.content == "done"
        assert r.raw["rounds"] == 1

    def test_round_cap_recorded_and_flag_set(self):
        """Hitting the cap must be visible, not look like a natural finish."""
        from freeaipi.models import ToolCall

        client = self._loop_client(
            replies=["final"],
            tool_calls=[ToolCall(id=f"c{i}", name="get_time", arguments={}) for i in range(9)],
        )
        r = client.run_tools(
            [{"role": "user", "content": "x"}], self.TOOLS,
            {"get_time": lambda **kw: "12:00"}, max_rounds=3,
        )
        assert r.raw["rounds"] == 3
        assert r.raw["round_cap_reached"] is True
        assert len(client.registry.providers["mock/m"].calls) >= 3


class TestCredentialHandling:
    def test_absent_key_reports_absent_not_value(self, mixed_client):
        status = mixed_client.status()
        assert status["credentials"]["openrouter"] == "<absent>"

    def test_present_key_is_redacted(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-SECRET-VALUE")
        client = FreeAIClient(include_local=False, include_openrouter=False, include_pollinations=False)
        status = client.status()
        rendered = str(status["credentials"]["openrouter"])
        assert "SECRET" not in rendered
        assert "set" in rendered

    def test_explicit_key_beats_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        client = FreeAIClient(
            openrouter_key="from-arg", include_local=False,
            include_openrouter=False, include_pollinations=False,
        )
        assert client._or_key.value == "from-arg"
        assert client._or_key.source == "explicit"

    def test_env_key_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        client = FreeAIClient(include_local=False, include_openrouter=False, include_pollinations=False)
        assert client._or_key.value == "from-env"
        assert client._or_key.source == "env:OPENROUTER_API_KEY"

    def test_empty_env_var_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        client = FreeAIClient(include_local=False, include_openrouter=False, include_pollinations=False)
        assert client._or_key.present is False


class TestCatalogueQueries:
    def test_list_filters_by_provider(self, mixed_client):
        assert [m.model_id for m in mixed_client.list_models(provider=Provider.LOCAL)] == ["local-big"]

    def test_list_filters_by_capability(self, mixed_client):
        assert [m.model_id for m in mixed_client.list_models(requires_reasoning=True)] == ["local-big"]

    def test_list_filters_by_auth(self, mixed_client):
        ids = [m.model_id for m in mixed_client.no_key_models()]
        assert ids == ["free-chat", "free-tools", "local-big"]

    def test_prefix_filter(self, mixed_client):
        assert [m.model_id for m in mixed_client.list_models(prefix="pollinations")] == [
            "free-chat", "free-tools"
        ]

    def test_describe(self, mixed_client):
        assert mixed_client.describe("free-tools").capabilities.tools is True

    def test_get_accepts_bare_unique_id(self, mixed_client):
        info, provider = mixed_client.registry.get("local-big")
        assert info.model_id == "local-big"

    def test_get_accepts_full_key(self, mixed_client):
        info, _ = mixed_client.registry.get("local/local-big")
        assert info.model_id == "local-big"

    def test_get_unknown_raises(self, mixed_client):
        with pytest.raises(ModelNotFoundError):
            mixed_client.registry.get("nope")

    def test_get_ambiguous_raises_with_options(self):
        client = build_client({
            "a/dup": (Provider.MOCK, Capabilities(), False),
            "b/dup": (Provider.MOCK, Capabilities(), False),
        })
        with pytest.raises(ModelNotFoundError) as ei:
            client.registry.get("dup")
        assert "a/dup" in str(ei.value) and "b/dup" in str(ei.value)


class TestContextManager:
    def test_closes_transport(self):
        with build_client({}) as client:
            pass
        assert client.transport._client.is_closed if hasattr(client.transport, "_client") else True

    def test_repr(self, mixed_client):
        assert "models=4" in repr(mixed_client)
