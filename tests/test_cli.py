"""CLI tests: argument parsing and the three subcommands."""

from __future__ import annotations

import json

import pytest

from freeaipi.cli import main
from freeaipi.errors import FreeAIPIError
from freeaipi.http import MockTransport
from freeaipi.models import Capabilities, ModelInfo, Provider
from freeaipi.providers.pollinations import PollinationsProvider


@pytest.fixture
def stub_client(monkeypatch):
    """Replace FreeAIClient with a stub so no network is touched."""
    import freeaipi.cli as cli

    class Stub:
        def __init__(self, **kwargs):
            self.registry = self._registry()
            self.warnings = ["stub warning"]

        def _registry(self):
            from freeaipi.registry import Registry

            reg = Registry()
            for mid, prov, caps, auth in (
                ("openai-fast", Provider.POLLINATIONS, Capabilities(text=True, tools=True), False),
                ("nemotron-3:free", Provider.OPENROUTER, Capabilities(text=True, tools=True), True),
            ):
                info = ModelInfo(
                    model_id=mid, provider=prov, provider_model_id=mid,
                    name=mid, capabilities=caps, auth_required=auth,
                )
                reg.register(info, PollinationsProvider(info, MockTransport()))
            return reg

        def list_models(self, **filters):
            return self.registry.list(**filters)

        def no_key_models(self):
            return self.registry.list(auth_required=False)

        def vendors(self):
            return self.registry.providers_in_use()

        def chat(self, message, **kwargs):
            from freeaipi.models import ChatResponse, Usage

            return ChatResponse(
                content=f"echo: {message}", model_id="openai-fast",
                provider=Provider.POLLINATIONS, usage=Usage(1, 1, 2), latency_ms=12.0,
            )

        def stream(self, message, **kwargs):
            from freeaipi.models import ChatChunk

            yield ChatChunk(delta="echo: ", model_id="openai-fast", provider=Provider.POLLINATIONS)
            yield ChatChunk(delta=message, model_id="openai-fast", provider=Provider.POLLINATIONS)

        def status(self):
            return {
                "models": 1,
                "providers": ["pollinations"],
                "no_key": ["pollinations/openai-fast"],
                "needs_key": [],
                "credentials": {"openrouter": "<absent>", "pollinations": "<absent>"},
                "warnings": self.warnings,
            }

        def close(self):
            pass

    monkeypatch.setattr(cli, "FreeAIClient", Stub)
    return Stub


class TestList:
    def test_lists_no_key_models_by_default(self, stub_client, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "pollinations/openai-fast" in out
        assert "openrouter" not in out  # keyed model hidden by default

    def test_no_key_only_flag_includes_keyed_models(self, stub_client, capsys):
        assert main(["list", "--no-key-only"]) == 0
        out = capsys.readouterr().out
        assert "openrouter/nemotron-3:free" in out

    def test_json_output_is_valid(self, stub_client, capsys):
        assert main(["list", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data[0]["model_id"] == "openai-fast"
        assert data[0]["free"] is True

    def test_provider_filter(self, stub_client, capsys):
        assert main(["list", "--provider", "openrouter"]) == 0
        out = capsys.readouterr().out
        assert "nemotron" not in out  # keyed, and hidden by default

    def test_vendor_filter(self, stub_client, capsys):
        assert main(["list", "--vendor", "pollinations"]) == 0
        out = capsys.readouterr().out
        assert "pollinations/openai-fast" in out
        assert "openrouter" not in out


class TestChat:
    def test_prints_content(self, stub_client, capsys):
        assert main(["chat", "hello"]) == 0
        assert "echo: hello" in capsys.readouterr().out

    def test_json_output(self, stub_client, capsys):
        assert main(["chat", "hello", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["content"] == "echo: hello"
        assert data["usage"]["cost_usd"] == 0.0
        assert data["provider"] == "pollinations"

    def test_stream_mode(self, stub_client, capsys):
        assert main(["chat", "hi", "--stream"]) == 0
        assert "echo: hi" in capsys.readouterr().out

    def test_no_message_is_error(self, stub_client, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin.read", lambda: "")
        assert main(["chat"]) == 2


class TestCheck:
    def test_status_json(self, stub_client, capsys):
        assert main(["check"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["providers"] == ["pollinations"]
        assert data["credentials"]["openrouter"] == "<absent>"


class TestErrorHandling:
    def test_library_error_exits_1(self, monkeypatch, capsys):
        import freeaipi.cli as cli

        class Boom:
            def __init__(self, **kwargs):
                raise FreeAIPIError("no credential configured")

        monkeypatch.setattr(cli, "FreeAIClient", Boom)
        assert main(["list"]) == 1
        assert "no credential" in capsys.readouterr().err
