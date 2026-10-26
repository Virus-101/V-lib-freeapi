"""Local providers: discovery, OpenAI-compat path, Ollama native path."""

from __future__ import annotations

import pytest

from freeaipi.http import MockTransport
from freeaipi.models import Provider
from freeaipi.providers.local import (
    LocalProvider,
    Usage_from_ollama,
    _infer_caps,
    build_models,
    discover,
)


class TestDiscovery:
    def test_discovers_ollama_and_openai_servers(self):
        """Two servers answering on different ports; dead ports skipped."""
        def fake_get(url, headers=None, params=None, timeout=None):
            if url.endswith("/api/tags"):
                return {"models": [{"name": "qwen2.5:7b"}, {"name": "llava"}]}
            if url.endswith("/models"):
                return {"data": [{"id": "local-model-1"}]}
            raise Exception("connection refused")

        transport = MockTransport()
        transport.get_json = fake_get
        found = discover(transport)
        kinds = {t.kind for t in found}
        assert "ollama" in kinds and "openai" in kinds
        ollama = [t for t in found if t.kind == "ollama"][0]
        assert "qwen2.5:7b" in ollama.models

    def test_all_ports_down_returns_empty(self):
        transport = MockTransport([Exception("refused") for _ in range(20)])
        assert discover(transport) == []

    def test_dead_port_skipped_not_fatal(self):
        """Ollama down must not stop the other servers from being discovered."""
        def fake_get(url, **kw):
            if "11434" in url:
                raise Exception("refused")
            if url.endswith("/models"):
                return {"data": [{"id": "m1"}]}
            raise Exception("refused")

        transport = MockTransport()
        transport.get_json = fake_get
        found = discover(transport)
        assert "ollama" not in {t.kind for t in found}
        assert "openai" in {t.kind for t in found}

    def test_extra_urls_probed(self):
        def fake_get(url, **kw):
            if "9999" in url:
                return {"data": [{"id": "custom-model"}]}
            raise Exception("refused")

        transport = MockTransport()
        transport.get_json = fake_get
        found = discover(transport, extra_urls=["http://127.0.0.1:9999/v1"])
        assert found[0].models == ("custom-model",)


class TestOpenAICompatPath:
    def test_chat_posts_to_chat_completions(self, local):
        transport, p = local([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        ])
        r = p.chat("hi")
        assert r.content == "OK"
        assert transport.requests[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
        assert r.provider is Provider.LOCAL
        assert r.tier == "local"

    def test_local_never_sends_authorization(self, local):
        transport, p = local([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert "Authorization" not in transport.requests[0]["headers"]

    def test_model_id_not_bare_name(self, local):
        """The wire id keeps the tag (``qwen2.5:7b``), not the bare name."""
        transport, p = local([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["json"]["model"] == "qwen2.5:7b"


class TestOllamaNativePath:
    def _ollama(self, responses):
        transport = MockTransport(responses)
        model = __import__("freeaipi.models", fromlist=["ModelInfo"]).ModelInfo(
            model_id="qwen2.5", provider=Provider.LOCAL,
            provider_model_id="qwen2.5:7b", name="qwen2.5",
        )
        return transport, LocalProvider(model, transport, "http://127.0.0.1:11434", kind="ollama")

    def test_uses_api_chat_not_openai_path(self):
        transport, p = self._ollama([
            {"message": {"content": "OK"}, "done": True, "done_reason": "stop",
             "prompt_eval_count": 10, "eval_count": 5}
        ])
        p.chat("hi")
        assert transport.requests[0]["url"] == "http://127.0.0.1:11434/api/chat"
        assert transport.requests[0]["json"]["stream"] is False

    def test_ollama_options_shape(self):
        transport, p = self._ollama([
            {"message": {"content": "OK"}, "done": True, "prompt_eval_count": 1, "eval_count": 1}
        ])
        p.chat("hi", temperature=0.2, max_tokens=100, seed=7, num_ctx=4096)
        body = transport.requests[0]["json"]
        assert body["options"]["temperature"] == 0.2
        assert body["options"]["num_predict"] == 100
        assert body["options"]["seed"] == 7
        assert body["options"]["num_ctx"] == 4096

    def test_ollama_tool_calls(self):
        transport, p = self._ollama([{
            "message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ]},
            "done": True, "prompt_eval_count": 1, "eval_count": 1,
        }])
        r = p.chat("weather")
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].arguments == {"city": "Paris"}

    def test_ollama_token_counts_mapped(self):
        transport, p = self._ollama([
            {"message": {"content": "OK"}, "done": True, "prompt_eval_count": 10, "eval_count": 5}
        ])
        r = p.chat("hi")
        assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens) == (10, 5, 15)

    def test_ollama_json_mode_sets_format(self):
        transport, p = self._ollama([
            {"message": {"content": "{}"}, "done": True, "prompt_eval_count": 1, "eval_count": 1}
        ])
        p.chat("hi", json_mode=True)
        assert transport.requests[0]["json"]["format"] == "json"

    def test_ollama_stream_ndjson(self):
        transport, p = self._ollama([[
            '{"message":{"content":"He"},"done":false}',
            '{"message":{"content":"llo"},"done":false}',
            '{"message":{},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":2}',
        ]])
        chunks = list(p.stream("hi"))
        assert "".join(c.delta for c in chunks if c.delta) == "Hello"
        assert chunks[-1].finish_reason == "stop"


class TestUsageMapping:
    def test_empty_when_no_counts(self):
        u = Usage_from_ollama({})
        assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (0, 0, 0)

    def test_only_completion_count(self):
        u = Usage_from_ollama({"eval_count": 7})
        assert u.completion_tokens == 7
        assert u.prompt_tokens == 0


class TestCapabilityInference:
    def test_context_from_name(self):
        assert _infer_caps("model-128k").context_length == 131072
        assert _infer_caps("llama3-8k").context_length == 8192

    def test_default_context(self):
        assert _infer_caps("some-model").context_length == 8192

    def test_vision_detected(self):
        assert _infer_caps("llava-13b").vision is True
        assert _infer_caps("qwen2.5-vl-7b").vision is True

    def test_reasoning_detected(self):
        assert _infer_caps("deepseek-r1-7b").reasoning is True
        assert _infer_caps("qwq-32b").reasoning is True

    def test_tools_detected_from_coder_suffix(self):
        assert _infer_caps("qwen2.5-coder:7b").tools is True


class TestModelBuilding:
    def test_registry_id_strips_tag(self):
        models = build_models(None, "http://127.0.0.1:11434", "ollama", ["qwen2.5:7b", "llava:latest"])
        assert [m.model_id for m in models] == ["qwen2.5", "llava"]
        # Wire id keeps the tag so the request reaches the right weights.
        assert [m.provider_model_id for m in models] == ["qwen2.5:7b", "llava:latest"]

    def test_free_and_no_auth(self):
        models = build_models(None, "u", "ollama", ["m"])
        assert models[0].free is True
        assert models[0].auth_required is False


class TestHealthCheck:
    def test_true_when_reachable(self, local):
        transport, p = local([{"data": []}])
        assert p.health_check() is True

    def test_false_when_down(self, local):
        transport, p = local([Exception("refused")])
        assert p.health_check() is False

    def test_ollama_health_uses_api_tags(self):
        transport = MockTransport([{"models": []}])
        model = __import__("freeaipi.models", fromlist=["ModelInfo"]).ModelInfo(
            model_id="m", provider=Provider.LOCAL, provider_model_id="m", name="m"
        )
        p = LocalProvider(model, transport, "http://127.0.0.1:11434", kind="ollama")
        assert p.health_check() is True
        assert transport.requests[0]["url"].endswith("/api/tags")
