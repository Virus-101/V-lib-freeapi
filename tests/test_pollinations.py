"""Pollinations provider: request shaping, response parsing, quota handling."""

from __future__ import annotations

import json

import pytest

from freeaipi.errors import ProviderError, RateLimitError
from freeaipi.http import FakeResponse
from freeaipi.models import Provider
from freeaipi.providers.pollinations import PollinationsProvider, build_models
from freeaipi.models import ModelInfo, Capabilities


class TestRequestShaping:
    def test_url_is_openai_path_not_root(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["url"] == "https://text.pollinations.ai/openai"

    def test_referrer_sent_by_default(self, pollinations):
        """The referrer lifts the anonymous quota; omitting it is a real regression."""
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["json"]["referrer"] == "freeaipi"

    def test_no_auth_header_without_key(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert "Authorization" not in transport.requests[0]["headers"]

    def test_auth_header_with_key(self, pollinations):
        transport = __import__("freeaipi.http", fromlist=["MockTransport"]).MockTransport([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        model = ModelInfo(
            model_id="openai-fast", provider=Provider.POLLINATIONS,
            provider_model_id="gpt-oss-20b", name="x",
        )
        p = PollinationsProvider(model, transport, api_key="pollen-abc")
        p.chat("hi")
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer pollen-abc"

    def test_system_message_prepended(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat([{"role": "user", "content": "hi"}], system="be terse")
        msgs = transport.requests[0]["json"]["messages"]
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert len(msgs) == 2

    def test_caller_messages_not_mutated(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        original = [{"role": "user", "content": "hi"}]
        p.chat(original, system="x")
        assert original == [{"role": "user", "content": "hi"}]

    def test_tools_and_tool_choice(self, pollinations):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        transport, p = pollinations([
            {"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}]}
        ])
        p.chat("hi", tools=tools)
        body = transport.requests[0]["json"]
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"

    def test_json_mode_sets_response_format(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi", json_mode=True)
        assert transport.requests[0]["json"]["response_format"] == {"type": "json_object"}

    def test_seed_forwarded(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi", seed=42)
        assert transport.requests[0]["json"]["seed"] == 42


class TestResponseParsing:
    def test_content_and_usage(self, pollinations):
        transport, p = pollinations([{
            "choices": [{"message": {"content": "hello", "reasoning": "because"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "user_tier": "anonymous",
        }])
        r = p.chat("hi")
        assert r.content == "hello"
        assert r.usage.total_tokens == 15
        assert r.reasoning == "because"
        assert r.tier == "anonymous"
        assert r.provider is Provider.POLLINATIONS

    def test_cost_is_zero(self, pollinations):
        transport, p = pollinations([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        assert p.chat("hi").cost_usd == 0.0

    def test_tool_calls_parsed(self, pollinations):
        transport, p = pollinations([{
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
            ]}, "finish_reason": "tool_calls"}]
        }])
        r = p.chat("weather?")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "get_weather"
        assert r.tool_calls[0].arguments == {"city": "Paris"}

    def test_malformed_tool_arguments_do_not_raise(self, pollinations):
        """A half-built argument string must surface, not crash the turn."""
        transport, p = pollinations([{
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "{not json"}}
            ]}, "finish_reason": "tool_calls"}]
        }])
        r = p.chat("go")
        assert r.tool_calls[0].arguments == {"_raw": "{not json"}

    def test_content_as_part_list_is_flattened(self, pollinations):
        transport, p = pollinations([{
            "choices": [{"message": {"content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}]},
                         "finish_reason": "stop"}]
        }])
        assert p.chat("hi").content == "abcd"

    def test_no_choices_raises_provider_error(self, pollinations):
        transport, p = pollinations([{"unexpected": True}])
        with pytest.raises(ProviderError):
            p.chat("hi")


class TestQuotaHandling:
    def test_http_402_raises_rate_limit_with_retry_after(self, pollinations):
        body = json.dumps({
            "status": 402, "success": False,
            "error": {"code": "FREE_LIMIT",
                      "message": "Free limit reached: 1 reply every 30 seconds per IP. Retry after 15s, or get unlimited replies with a pollen key at https://enter.pollinations.ai/?ref=classic"},
        })
        transport, p = pollinations([FakeResponse(402, text=body)])
        with pytest.raises(RateLimitError) as ei:
            p.chat("hi")
        assert ei.value.retry_after == 15.0

    def test_402_without_retry_hint_still_raises(self, pollinations):
        transport, p = pollinations([FakeResponse(402, text=json.dumps({"status": 402, "error": {"message": "quota"}}))])
        with pytest.raises(RateLimitError) as ei:
            p.chat("hi")
        assert ei.value.retry_after is None

    def test_other_4xx_raises_provider_error(self, pollinations):
        transport, p = pollinations([FakeResponse(404, text="nope")])
        with pytest.raises(ProviderError) as ei:
            p.chat("hi")
        assert ei.value.status_code == 404

    def test_non_json_2xx_raises_provider_error(self, pollinations):
        transport, p = pollinations([FakeResponse(200, text="<html>oops</html>")])
        with pytest.raises(ProviderError):
            p.chat("hi")


class TestStreaming:
    def test_deltas_yielded_as_text(self, pollinations):
        frames = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ]
        transport, p = pollinations([frames])
        chunks = list(p.stream("hi"))
        text = "".join(c.delta for c in chunks if c.delta)
        assert text == "Hello"

    def test_reasoning_yielded_separately(self, pollinations):
        frames = [
            'data: {"choices":[{"delta":{"reasoning":"think"}}]}',
            'data: {"choices":[{"delta":{"content":"answer"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ]
        transport, p = pollinations([frames])
        chunks = list(p.stream("hi"))
        assert "".join(c.reasoning for c in chunks if c.reasoning) == "think"
        assert "".join(c.delta for c in chunks if c.delta) == "answer"

    def test_tool_call_fragments_accumulate(self, pollinations):
        frames = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"Paris\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            'data: [DONE]',
        ]
        transport, p = pollinations([frames])
        chunks = list(p.stream("hi"))
        calls = [c.tool_call for c in chunks if c.tool_call]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"] == {"city": "Paris"}

    def test_usage_only_frames_are_skipped(self, pollinations):
        """A frame with no choices carries accounting, not text."""
        frames = [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]',
        ]
        transport, p = pollinations([frames])
        chunks = list(p.stream("hi"))
        assert "".join(c.delta for c in chunks if c.delta) == "hi"

    def test_unparseable_lines_are_skipped(self, pollinations):
        frames = [
            "data: {broken",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        transport, p = pollinations([frames])
        assert "".join(c.delta for c in p.stream("hi") if c.delta) == "ok"


class TestDiscovery:
    def test_build_models_reads_endpoint(self):
        from freeaipi.http import MockTransport
        transport = MockTransport([[{
            "name": "openai-fast", "description": "d", "reasoning": True, "tier": "anonymous",
            "community": False, "input_modalities": ["text"], "output_modalities": ["text"],
            "tools": True, "aliases": ["openai", "gpt-oss"],
        }]])
        models = build_models(transport)
        assert len(models) == 1
        assert models[0].model_id == "openai-fast"
        assert models[0].capabilities.tools is True
        assert models[0].capabilities.reasoning is True
        assert models[0].auth_required is False

    def test_build_models_falls_back_on_failure(self):
        from freeaipi.http import MockTransport
        transport = MockTransport([Exception("network down")])
        models = build_models(transport)
        assert len(models) == 1
        assert models[0].model_id == "openai-fast"

    def test_build_models_ignores_non_anonymous(self):
        from freeaipi.http import MockTransport
        transport = MockTransport([[{"name": "paid-model", "tier": "plus", "tools": True}]])
        models = build_models(transport)
        # Still registered: the tier field is not used to filter here.
        assert models[0].model_id == "paid-model"
