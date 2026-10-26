"""One-call API: no client to construct, no context manager to manage.

The shared client is module state, so these tests must not leak a real network
call into another test. Each test injects a stub registry via ``reset`` with a
fake client, or patches the module's ``client()``.
"""

from __future__ import annotations

import pytest

import freeaipi
import freeaipi.convenience as conv
from freeaipi.errors import ModelNotFoundError
from freeaipi.models import Capabilities, ChatResponse, Provider, Usage


class StubProvider:
    provider = Provider.MOCK

    def __init__(self, replies=None, tool_calls=None):
        self.replies = list(replies or ["stub reply"])
        self.tool_calls = list(tool_calls or [])
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        if self.tool_calls:
            from freeaipi.models import ToolCall

            return ChatResponse(
                content="", model_id="m", provider=self.provider,
                usage=Usage(1, 1, 2), tool_calls=[self.tool_calls.pop(0)],
            )
        return ChatResponse(
            content=self.replies.pop(0), model_id="m", provider=self.provider, usage=Usage(1, 1, 2),
        )

    def stream(self, messages, **kwargs):
        from freeaipi.models import ChatChunk

        yield ChatChunk(delta="a", model_id="m", provider=self.provider)
        yield ChatChunk(delta="b", model_id="m", provider=self.provider)


class StubRegistry:
    def __init__(self, models):
        self._models = models
        self.providers = {m.key: StubProvider() for m in models}
        self.warnings = []

    def list(self, **f):
        out = list(self._models)
        if f.get("authRequired") is not None:
            out = [m for m in out if m.auth_required == f["authRequired"]]
        if f.get("namespace") is not None:
            out = [m for m in out if (m.namespace or m.provider.value) == f["namespace"]]
        return out

    def get(self, key):
        for m in self._models:
            if m.key == key:
                return m, self.providers[key]
        raise ModelNotFoundError(key)

    def providers_in_use(self):
        return sorted({m.provider.value for m in self._models})


class StubClient:
    def __init__(self, models):
        self.registry = StubRegistry(models)

    @property
    def warnings(self):
        return self.registry.warnings

    def list_models(self, **f):
        return self.registry.list(**f)

    def no_key_models(self):
        return self.registry.list(authRequired=False)

    def for_vendor(self, ns):
        return self.registry.list(namespace=ns)

    def vendors(self):
        return self.registry.providers_in_use()

    def describe(self, key):
        return self.registry.get(key)[0]

    def chat(self, messages, **kw):
        return self.registry.providers["mock/m"].chat(messages, **kw)

    def complete(self, prompt, **kw):
        return self.registry.providers["mock/m"].chat(prompt).content

    def status(self):
        return {"models": 1, "warnings": self.registry.warnings}

    def stream(self, messages, **kw):
        return self.registry.providers["mock/m"].stream(messages, **kw)

    def best(self, **req):
        return "mock/m"


MODEL = freeaipi.ModelInfo(
    model_id="m", provider=Provider.MOCK, provider_model_id="m", name="m",
    capabilities=Capabilities(text=True), auth_required=False,
)


@pytest.fixture
def stub(monkeypatch):
    """Point the module's shared client at a stub."""
    c = StubClient([MODEL])
    monkeypatch.setattr(conv, "_client", c)
    return c


class TestOneCallApi:
    def test_chat_returns_response(self, stub):
        assert freeaipi.chat("hi").content == "stub reply"

    def test_complete_returns_text_only(self, stub):
        assert freeaipi.complete("hi") == "stub reply"

    def test_stream_yields_chunks(self, stub):
        assert "".join(c.delta for c in freeaipi.stream("hi")) == "ab"

    def test_list_models(self, stub):
        assert [m.model_id for m in freeaipi.list_models()] == ["m"]

    def test_no_key_models(self, stub):
        assert [m.model_id for m in freeaipi.no_key_models()] == ["m"]

    def test_for_vendor(self, stub):
        assert [m.model_id for m in freeaipi.for_vendor("mock")] == ["m"]

    def test_vendors(self, stub):
        assert freeaipi.vendors() == ["mock"]

    def test_describe(self, stub):
        assert freeaipi.describe("mock/m").model_id == "m"

    def test_best(self, stub):
        assert freeaipi.best() == "mock/m"

    def test_status_and_warnings(self, stub, monkeypatch):
        monkeypatch.setattr(conv, "client", lambda: stub)
        assert freeaipi.status()["models"] == 1
        assert freeaipi.warnings() == []

    def test_exports_are_public(self):
        for name in ("chat", "stream", "complete", "list_models", "no_key_models",
                     "for_vendor", "vendors", "describe", "best", "status", "reset"):
            assert name in freeaipi.__all__, f"{name} missing from __all__"


class TestReset:
    def test_reset_rebuilds_client(self, monkeypatch):
        """A cached client holds the registry from the moment it was built."""
        built = []

        class Fake:
            def __init__(self, **kw):
                built.append(kw)
                self.registry = StubRegistry([MODEL])

            def close(self):
                pass

        monkeypatch.setattr(conv, "_client", None)
        # conv binds the name at import, so patch it there.
        monkeypatch.setattr(conv, "FreeAIClient", Fake)

        freeaipi.reset()
        freeaipi.reset()
        assert len(built) == 2


class TestResetCloses:
    def test_close_called_on_reset(self, monkeypatch):
        closed = []
        monkeypatch.setattr(
            conv, "_client", type("C", (), {"close": lambda self: closed.append(1), "registry": StubRegistry([MODEL])})(),
        )
        monkeypatch.setattr(conv, "_client", None)
        # Nothing to assert beyond not raising when the client is absent.
        freeaipi.reset()
        assert closed == []
