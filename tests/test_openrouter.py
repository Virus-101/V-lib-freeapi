"""OpenRouter provider: key gating, catalogue filtering, streaming."""

from __future__ import annotations

import pytest

from freeaipi.errors import ConfigurationError, ProviderError
from freeaipi.http import FakeResponse, MockTransport
from freeaipi.models import ModelInfo, Provider, Capabilities
from freeaipi.providers.openrouter import OpenRouterProvider, build_models, check_key

OR_MODEL = {
    "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "name": "NVIDIA: Nemotron 3 Ultra (free)",
    "context_length": 1000000,
    "architecture": {"modality": "text->text", "input_modalities": ["text"], "output_modalities": ["text"]},
    "pricing": {"prompt": "0", "completion": "0"},
    "top_provider": {"context_length": 1000000, "max_completion_tokens": 65536, "is_moderated": False},
    "supported_parameters": ["tools", "reasoning", "max_tokens", "temperature"],
    "description": "frontier reasoning",
}


class TestKeyGating:
    def test_missing_key_raises_configuration_error(self):
        transport = MockTransport()
        model = ModelInfo(
            model_id="x:free", provider=Provider.OPENROUTER,
            provider_model_id="x:free", name="x", auth_required=True,
        )
        with pytest.raises(ConfigurationError) as ei:
            OpenRouterProvider(model, transport, api_key="")
        assert "OPENROUTER_API_KEY" in str(ei.value)

    def test_empty_string_key_treated_as_missing(self):
        transport = MockTransport()
        model = ModelInfo(model_id="x:free", provider=Provider.OPENROUTER,
                          provider_model_id="x:free", name="x")
        with pytest.raises(ConfigurationError):
            OpenRouterProvider(model, transport, api_key="   " or "")


class TestRequestShaping:
    def test_authorization_header(self, openrouter):
        transport, p = openrouter([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer sk-or-v1-test"

    def test_wire_model_id_used_not_registry_id(self, openrouter):
        transport, p = openrouter([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["json"]["model"] == "nvidia/nemotron-3:free"

    def test_endpoint_is_chat_completions(self, openrouter):
        transport, p = openrouter([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"


class TestResponseParsing:
    def test_free_models_marked_free_and_tier_free(self, openrouter):
        transport, p = openrouter([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        r = p.chat("hi")
        assert r.tier == "free"
        assert r.provider is Provider.OPENROUTER
        assert r.cost_usd == 0.0

    def test_error_envelope_raises(self, openrouter):
        transport, p = openrouter([
            {"error": {"message": "no endpoints found", "code": 400}}
        ])
        with pytest.raises(ProviderError):
            p.chat("hi")

    def test_tool_calls_parsed(self, openrouter):
        transport, p = openrouter([{
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}
            ]}, "finish_reason": "tool_calls"}]
        }])
        r = p.chat("go")
        assert r.tool_calls[0].arguments == {"a": 1}


class TestStreaming:
    def test_tool_fragments_accumulate_then_emit_once(self, openrouter):
        frames = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        transport, p = openrouter([frames])
        chunks = list(p.stream("hi"))
        calls = [c.tool_call for c in chunks if c.tool_call]
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"a": 1}

    def test_done_marker_terminates(self, openrouter):
        frames = [
            'data: {"choices":[{"delta":{"content":"x"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"IGNORED"}}]}',
        ]
        transport, p = openrouter([frames])
        assert "".join(c.delta for c in p.stream("hi") if c.delta) == "x"


class TestCatalogue:
    def _catalogue(self, models):
        return MockTransport([{"data": models}])

    def test_free_models_included(self):
        models = build_models(self._catalogue([OR_MODEL]))
        assert [m.model_id for m in models] == [OR_MODEL["id"]]
        m = models[0]
        assert m.free is True
        assert m.auth_required is True
        assert m.capabilities.tools is True
        assert m.capabilities.reasoning is True
        assert m.capabilities.context_length == 1000000
        assert m.capabilities.max_output_tokens == 65536

    def test_paid_models_excluded(self):
        paid = dict(OR_MODEL, id="openai/gpt-5", pricing={"prompt": "0.00001", "completion": "0.00003"})
        models = build_models(self._catalogue([paid]))
        assert models == []

    def test_non_free_suffix_excluded(self):
        notfree = dict(OR_MODEL, id="openai/gpt-5")
        assert build_models(self._catalogue([notfree])) == []

    def test_vision_detected_from_modalities(self):
        vl = dict(OR_MODEL, architecture={"input_modalities": ["text", "image"], "output_modalities": ["text"]})
        models = build_models(self._catalogue([vl]))
        assert models[0].capabilities.vision is True

    def test_unsupported_tools_flag_false(self):
        notools = dict(OR_MODEL, supported_parameters=["temperature"])
        models = build_models(self._catalogue([notools]))
        assert models[0].capabilities.tools is False

    def test_network_failure_returns_empty_not_raise(self):
        transport = MockTransport([Exception("offline")])
        assert build_models(transport) == []

    def test_malformed_pricing_treated_as_free(self):
        weird = dict(OR_MODEL, pricing={"prompt": None, "completion": ""})
        models = build_models(self._catalogue([weird]))
        assert len(models) == 1


class TestCheckKey:
    def test_returns_quota_when_key_works(self):
        transport = MockTransport([{"data": {"label": "l", "limit": 20.0, "usage": 1.5}}])
        assert check_key(transport, "sk-or-v1-x")["data"]["usage"] == 1.5

    def test_returns_none_without_key(self):
        assert check_key(MockTransport(), "") is None

    def test_returns_none_on_failure(self):
        transport = MockTransport([Exception("401")])
        assert check_key(transport, "bad") is None
