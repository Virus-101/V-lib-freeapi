"""OpenRouter's ``:free`` tier.

``GET /api/v1/models`` is public and returns the full catalog with pricing; the
free models are the ones whose id ends in ``:free`` and whose ``pricing`` is
zero. ``POST /api/v1/chat/completions`` requires an account key — the free tier
is free *money*, not free *account*. ``GET /api/v1/key`` returns the remaining
quota and is called once per client construction so a dead key fails fast
instead of on the first chat.

As of 2026-10-26 the free catalog holds 17 models, including
``nvidia/nemotron-3-ultra-550b-a55b:free`` (1M context), ``poolside/laguna-xs-2.1:free``
and ``cohere/north-mini-code:free``. That list changes weekly, which is why it is
fetched rather than hardcoded.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from .base import BaseProvider
from ..errors import ConfigurationError, ProviderError
from ..models import ChatChunk, ChatResponse, ModelInfo, Provider, ToolCall

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    provider = Provider.OPENROUTER

    def __init__(self, model: ModelInfo, transport, api_key: str, base_url: str = BASE_URL):
        super().__init__(model, transport)
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                "openrouter requires an API key. Get a free one at https://openrouter.ai/keys "
                "and set OPENROUTER_API_KEY."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Virus-101/V-lib-freeapi",
            "X-Title": "freeaipi",
        }

    def _payload(self, messages, tools, temperature, max_tokens, json_mode, seed, stream=False):
        payload: dict[str, Any] = {
            "model": self.model.provider_model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        if stream:
            payload["stream"] = True
        return payload

    def chat(
        self,
        messages,
        tools=None,
        temperature=0.7,
        max_tokens=2048,
        system=None,
        json_mode=False,
        seed=None,
        **kwargs: Any,
    ) -> ChatResponse:
        started = time.time()
        msgs = self._merge_system(messages, system)
        payload = self._payload(msgs, tools, temperature, max_tokens, json_mode, seed)
        data = self.transport.post_json(
            f"{self.base_url}/chat/completions", payload, self._headers()
        )
        return self._to_response(data, started)

    def _to_response(self, data: dict[str, Any], started: float) -> ChatResponse:
        if "error" in data:
            raise ProviderError(f"OpenRouter error: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"OpenRouter returned no choices: {str(data)[:200]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return ChatResponse(
            content=content,
            model_id=self.model.model_id,
            provider=self.provider,
            usage=self._usage_from(data),
            tool_calls=[ToolCall.from_openai(tc) for tc in (msg.get("tool_calls") or [])],
            finish_reason=choice.get("finish_reason") or "stop",
            latency_ms=(time.time() - started) * 1000,
            reasoning=msg.get("reasoning") or "",
            raw=data,
            tier="free",
        )

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=2048, system=None,
               json_mode=False, seed=None, **kwargs: Any) -> Iterable[ChatChunk]:
        msgs = self._merge_system(messages, system)
        payload = self._payload(msgs, tools, temperature, max_tokens, json_mode, seed, stream=True)
        model_id = self.model.model_id

        pending: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        for line in self.transport.stream_lines(
            f"{self.base_url}/chat/completions", payload, self._headers()
        ):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = __import__("json").loads(chunk)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                yield ChatChunk(delta=delta["content"], model_id=model_id, provider=self.provider)
            if delta.get("reasoning"):
                yield ChatChunk(reasoning=delta["reasoning"], model_id=model_id, provider=self.provider)
            for tc in delta.get("tool_calls") or []:
                slot = pending.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            finish = choices[0].get("finish_reason")
            if finish:
                for slot in pending.values():
                    if slot["name"]:
                        yield ChatChunk(
                            tool_call={"name": slot["name"], "arguments": _safe_json(slot["arguments"])},
                            finish_reason=finish,
                            model_id=model_id,
                            provider=self.provider,
                        )
                pending.clear()
                yield ChatChunk(
                    finish_reason=finish,
                    usage=self._usage_from({"usage": usage}),
                    model_id=model_id,
                    provider=self.provider,
                )


def _safe_json(text: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(text or "{}")
        return parsed if isinstance(parsed, dict) else {"_raw": text}
    except (ValueError, TypeError):
        return {"_raw": text}


def build_models(transport, api_key: str = "") -> list[ModelInfo]:
    """Fetch the ``:free`` tier.

    Returns an empty list if the catalog cannot be reached — the registry then
    simply has no OpenRouter entries, which is correct behaviour rather than a
    crash at import time.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        data = transport.get_json(f"{BASE_URL}/models", headers)
    except Exception:
        return []

    out: list[ModelInfo] = []
    for m in (data or {}).get("data") or []:
        mid = m.get("id") or ""
        if not mid.endswith(":free"):
            continue
        pricing = m.get("pricing") or {}
        try:
            paid = float(pricing.get("prompt") or 0) > 0 or float(pricing.get("completion") or 0) > 0
        except (TypeError, ValueError):
            paid = False
        if paid:
            continue
        arch = m.get("architecture") or {}
        in_mods = arch.get("input_modalities") or ["text"]
        supported = m.get("supported_parameters") or []
        top = m.get("top_provider") or {}
        out.append(
            ModelInfo(
                model_id=mid,
                provider=Provider.OPENROUTER,
                provider_model_id=mid,
                name=m.get("name") or mid,
                capabilities=_caps_from(m, in_mods, supported, top),
                free=True,
                auth_required=True,
                rate_limit="Free-tier quota per key; see GET /api/v1/key",
                notes=m.get("description") or "",
            )
        )
    return out


def _caps_from(m: dict[str, Any], in_mods: list[str], supported: list[str], top: dict[str, Any]) -> Any:
    from ..models import Capabilities

    return Capabilities(
        text=True,
        vision="image" in in_mods,
        tools="tools" in supported,
        reasoning="reasoning" in supported or "include_reasoning" in supported,
        json_mode="response_format" in supported or "tools" in supported,
        context_length=m.get("context_length") or top.get("context_length") or 32768,
        max_output_tokens=top.get("max_completion_tokens") or 4096,
    )


def check_key(transport, api_key: str) -> dict[str, Any] | None:
    """Return the key's quota info, or ``None`` if it is not usable.

    Called at client construction so a bad key is reported immediately.
    """
    if not api_key:
        return None
    try:
        return transport.get_json(f"{BASE_URL}/key", {"Authorization": f"Bearer {api_key}"})
    except Exception:
        return None
