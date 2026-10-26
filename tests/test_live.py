"""Live integration tests against the real free endpoints.

These hit the network and are skipped by default. Run them with:

    LIVE=1 py -3.12 -m pytest tests/test_live.py -v

They exist because a mock cannot tell you a provider changed its wire format,
and two of the behaviours that matter most are invisible offline:

* the anonymous quota (HTTP 402 with a retry hint), and
* whether a ``referrer`` actually lifts it.

Both were discovered by calling the real endpoint, and both are asserted here.
"""

from __future__ import annotations

import os

import pytest

from freeaipi import FreeAIClient
from freeaipi.errors import RateLimitError
from freeaipi.models import Provider

pytestmark = [
    pytest.mark.skipif(os.environ.get("LIVE") != "1", reason="needs LIVE=1 (hits the network)"),
    pytest.mark.live,
]


@pytest.fixture(scope="module")
def client():
    with FreeAIClient(include_local=False) as c:
        yield c


class TestCatalogue:
    def test_pollinations_model_discovered(self, client):
        models = [m for m in client.list_models() if m.provider is Provider.POLLINATIONS]
        assert models, "Pollinations model missing from registry"

    def test_pollinations_needs_no_key(self, client):
        m = client.list_models(provider=Provider.POLLINATIONS)[0]
        assert m.auth_required is False

    def test_openrouter_models_gated_on_key(self, client):
        """Without a key the registry holds no OpenRouter entries.

        Deliberate: a catalogue of unusable models is worse than an honest
        warning that says how many exist and what unlocks them.
        """
        models = [m for m in client.list_models() if m.provider is Provider.OPENROUTER]
        assert models == []
        assert any("OPENROUTER_API_KEY" in w for w in client.warnings)

    def test_openrouter_catalogue_fetchable_without_key(self):
        """The catalogue endpoint is public; only *calling* needs a key."""
        from freeaipi.http import Transport
        from freeaipi.providers.openrouter import build_models

        models = build_models(Transport())
        assert len(models) >= 10, f"only {len(models)} OpenRouter free models seen"
        assert all(m.free for m in models)
        assert all(m.auth_required for m in models)

    def test_openrouter_free_models_have_context_lengths(self):
        from freeaipi.http import Transport
        from freeaipi.providers.openrouter import build_models

        models = build_models(Transport())
        assert all(m.capabilities.context_length > 0 for m in models)
        assert any(m.capabilities.tools for m in models)

    def test_every_model_is_free(self, client):
        assert all(m.free for m in client.list_models())

    def test_capabilities_populated(self, client):
        m = client.list_models(provider=Provider.POLLINATIONS)[0]
        assert m.capabilities.context_length > 0
        assert m.capabilities.text is True


class TestLivePollinations:
    def test_returns_content(self, client):
        r = client.chat("Reply with exactly: OK", model="pollinations/openai-fast", max_tokens=64)
        assert r.content.strip() == "OK"

    def test_reports_anonymous_tier(self, client):
        r = client.chat("say hi", model="pollinations/openai-fast", max_tokens=16)
        assert r.tier == "anonymous"

    def test_reports_usage_and_zero_cost(self, client):
        r = client.chat("say hi", model="pollinations/openai-fast", max_tokens=16)
        assert r.usage.total_tokens > 0
        assert r.cost_usd == 0.0

    def test_reasoning_field_is_separate_from_content(self, client):
        r = client.chat("2+2?", model="pollinations/openai-fast", max_tokens=256)
        assert r.content.strip() != ""
        # Reasoning is a separate field; it must not be glued onto the answer.
        assert "The user" not in r.content or r.reasoning

    def test_streaming_yields_text(self, client):
        chunks = list(client.stream("Count: 1 2 3", model="pollinations/openai-fast", max_tokens=64))
        text = "".join(c.delta for c in chunks if c.delta)
        assert text.strip(), "stream produced no text"

    def test_tool_call_is_parsed(self, client):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        r = client.chat(
            "What is the weather in Paris? You must use the get_weather tool.",
            model="pollinations/openai-fast", tools=tools, max_tokens=400,
        )
        assert r.tool_calls, f"no tool call; content was {r.content!r}"
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].arguments.get("city")

    def test_quota_error_carries_retry_hint(self):
        """Hammer the endpoint to trigger the anonymous 402, then check the hint.

        The referrer usually keeps this from firing, so a skip is the honest
        outcome when the limit is not reached — the assertion is about the
        *shape* of the error when it is.
        """
        with FreeAIClient(include_local=False) as c:
            triggered = False
            for _ in range(6):
                try:
                    c.chat("hi", model="pollinations/openai-fast", max_tokens=8)
                except RateLimitError as exc:
                    assert exc.retry_after is not None, "quota error lacked a retry hint"
                    assert exc.retry_after >= 0
                    triggered = True
                    break
            if not triggered:
                pytest.skip("anonymous quota never triggered (referrer lifted it)")
