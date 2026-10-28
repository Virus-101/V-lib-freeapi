"""Per-developer state: key store, usage ledger, short alias, local gateway."""

import json
import threading
import urllib.request

import pytest

from freeaipi import home
from freeaipi.cli import main
from freeaipi.client import FreeAIClient
from freeaipi.gateway import make_server
from freeaipi.http import FakeResponse, MockTransport


def test_set_and_read_key_roundtrip():
    home.set_key("GROQ_API_KEY", "gsk_test_123")
    assert home.read_keys() == {"GROQ_API_KEY": "gsk_test_123"}
    assert home.remove_key("GROQ_API_KEY") is True
    assert home.read_keys() == {}
    assert home.remove_key("GROQ_API_KEY") is False


def test_stored_key_does_not_override_env(monkeypatch):
    home.set_key("GROQ_API_KEY", "from-file")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert home.load_keys() == []
    import os
    assert os.environ["GROQ_API_KEY"] == "from-env"


def test_stored_key_fills_missing_env(monkeypatch):
    home.set_key("GROQ_API_KEY", "from-file")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert home.load_keys() == ["GROQ_API_KEY"]


def test_key_list_never_prints_values(capsys):
    home.set_key("GROQ_API_KEY", "super-secret-value")
    assert main(["key", "list"]) == 0
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "GROQ_API_KEY" in out and "stored" in out


def test_usage_summary_counts_calls():
    home.record_usage("groq/a", "groq", 3, 4, 7)
    home.record_usage("groq/a", "groq", 1, 1, 2)
    home.record_usage("local/b", "local", 0, 0, 5)
    s = home.usage_summary()
    assert s["calls"] == 3 and s["tokens"] == 14
    assert s["models"]["groq/a"] == {"calls": 2, "tokens": 9}


def test_usage_since_filters_old_rows():
    home.record_usage("m", "p", 1, 1, 2)
    assert home.usage_summary(since=4_102_444_800)["calls"] == 0


def test_chat_writes_ledger():
    from freeaipi.models import Provider
    from freeaipi.response import ChatResponse, Usage

    class Prov:
        def chat(self, **kw):
            return ChatResponse(content="OK", model_id="x", provider=Provider.LOCAL,
                                usage=Usage(2, 1, 3))

    class Reg:
        models = {"x": 1}

        def get(self, key):
            return object(), Prov()

    c = FreeAIClient.__new__(FreeAIClient)
    c.registry = Reg()
    c.chat("hi", model="x")
    s = home.usage_summary()
    assert s["calls"] == 1 and s["tokens"] == 3
    assert s["models"]["x"]["calls"] == 1


def test_fllm_is_an_alias_for_freeaipi():
    import fllm
    import freeaipi

    assert fllm.chat is freeaipi.chat
    assert fllm.__version__ == freeaipi.__version__
    assert fllm.FreeAIClient is freeaipi.FreeAIClient


def test_unknown_key_vendor_is_rejected(capsys):
    assert main(["key", "set", "nonsense"]) == 2
    assert "unknown vendor" in capsys.readouterr().err


def test_gateway_lists_models_and_rejects_bad_body():
    class Stub:
        registry = [1, 2]

        def list_models(self):
            class M:
                key, label = "groq/x", "groq"
            return [M()]

        def chat(self, messages, **kw):
            raise AssertionError("not reached")

    srv = make_server(Stub(), port=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models") as r:
            data = json.load(r)
        assert data["data"][0]["id"] == "groq/x"
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_gateway_chat_completion_shape():
    from freeaipi.models import Provider
    from freeaipi.response import ChatResponse, Usage

    class Stub:
        registry = []

        def list_models(self):
            return []

        def chat(self, messages, **kw):
            return ChatResponse(content="hello", model_id="local/x", provider=Provider.LOCAL,
                                usage=Usage(1, 2, 3))

    srv = make_server(Stub(), port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
            data=json.dumps({"model": "auto", "messages": [{"role": "user", "content": "hi"}]}).encode())
        with urllib.request.urlopen(req) as r:
            body = json.load(r)
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["usage"]["total_tokens"] == 3
    finally:
        srv.shutdown()
        srv.server_close()
