"""Generic OpenAI-compatible provider.

Nearly every LLM vendor speaks the same wire protocol — ``POST
{base_url}/chat/completions`` with an ``Authorization: Bearer`` header, plus
``GET {base_url}/models`` for discovery. Hardcoding a class per vendor is busy
work that produces identical code fifteen times.

So: one provider, driven by a :class:`~freeaipi.presets.ProviderSpec`. Adding a
free tier is a config entry, not a library change. That is the difference
between "covers the providers I got round to" and "covers the market".

The ``Provider.COMPAT`` kind stays stable for dispatch while
:attr:`~freeaipi.models.ModelInfo.namespace` carries the vendor name, so the
registry key reads ``groq/llama-3.3-70b-versatile``.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from .base import BaseProvider
from ..errors import ConfigurationError, ProviderError
from ..models import Capabilities, ChatChunk, ChatResponse, ModelInfo, Provider, ToolCall

#: Extra headers worth sending to the aggregators that rank referrals.
DEFAULT_HEADERS = {"X-Title": "freeaipi", "HTTP-Referer": "https://github.com/Virus-101/freeaipi"}


class OpenAICompatProvider(BaseProvider):
    """Any OpenAI-compatible endpoint.

    ``spec`` supplies the base url, the credential, and how to build model
    entries. A ``models`` list in the spec is used verbatim; otherwise the
    catalogue is fetched from ``{base_url}/models``.
    """

    provider = Provider.COMPAT

    def __init__(self, model: ModelInfo, transport, spec, base_url: str, api_key: str):
        super().__init__(model, transport)
        # Only demand a key when the spec declares one. A private vLLM or a
        # local endpoint legitimately needs no auth, and refusing it here would
        # make the generic provider unusable for exactly the case it is for.
        if spec.key_env and not api_key:
            raise ConfigurationError(
                f"{spec.namespace} requires an API key "
                f"({spec.key_env}). Set it, or remove it from the enabled providers."
            )
        self.spec = spec
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=2048, system=None,
             json_mode=False, seed=None, **kwargs: Any) -> ChatResponse:
        started = time.time()
        msgs = self._merge_system(messages, system)
        payload = self._payload(msgs, tools, temperature, max_tokens, json_mode, seed)
        data = self.transport.post_json(
            f"{self.base_url}/chat/completions", payload, self._headers()
        )
        return self._to_response(data, started)

    def _to_response(self, data: dict[str, Any], started: float) -> ChatResponse:
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError(f"{self.model.key} error: {data['error']}")
        choices = (data or {}).get("choices") or []
        if not choices:
            raise ProviderError(f"{self.model.key} returned no choices: {str(data)[:200]}")
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
            tier="free" if self.model.free else "",
        )

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=2048, system=None,
               json_mode=False, seed=None, **kwargs: Any) -> Iterable[ChatChunk]:
        msgs = self._merge_system(messages, system)
        payload = self._payload(msgs, tools, temperature, max_tokens, json_mode, seed, stream=True)
        model_id = self.model.model_id

        import json

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
                obj = json.loads(chunk)
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


# ── catalogue parsing ────────────────────────────────────────────────────────

def parse_catalogue(raw: Any, spec) -> list[ModelInfo]:
    """Turn a ``/models`` response into registry entries.

    Handles the two shapes in the wild: OpenAI's ``{"data": [...]}`` and a bare
    list. Anything unparseable is skipped rather than aborting the whole
    provider, because one odd entry should not hide fifty good models.
    """
    entries = []
    if isinstance(raw, dict):
        entries = raw.get("data") or raw.get("models") or []
    elif isinstance(raw, list):
        entries = raw

    out: list[ModelInfo] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        mid = e.get("id") or e.get("name") or e.get("slug")
        if not mid:
            continue
        ctx = (
            e.get("context_length")
            or (e.get("top_provider") or {}).get("context_length")
            or 32768
        )
        max_out = (
            e.get("max_completion_tokens")
            or (e.get("top_provider") or {}).get("max_completion_tokens")
            or min(8192, ctx // 4)
        )
        out.append(
            ModelInfo(
                model_id=mid,
                provider=Provider.COMPAT,
                provider_model_id=mid,
                name=e.get("name") or mid,
                capabilities=_caps_from(e, ctx, max_out),
                free=spec.free,
                auth_required=True,
                rate_limit=spec.rate_limit or "",
                notes=spec.notes,
                namespace=spec.namespace,
            )
        )
    return out


def _caps_from(e: dict[str, Any], ctx: int, max_out: int) -> Capabilities:
    """Read capabilities from the catalogue entry, with name-based fallback.

    Aggregators differ on which fields they publish. Where the field is absent
    the name is consulted, because a model called ``llama-3.3-70b`` obviously
    does text even if the entry omits the modality list.
    """
    in_mods = (
        ((e.get("architecture") or {}).get("input_modalities"))
        or e.get("input_modalities")
        or ["text"]
    )
    supported = e.get("supported_parameters") or []
    name = (e.get("id") or e.get("name") or "").lower()
    return Capabilities(
        text=True,
        vision="image" in in_mods or "vl" in name or "vision" in name or "llava" in name,
        tools="tools" in supported or "tool" in name or "instruct" in name or "coder" in name,
        reasoning="reasoning" in supported or "reason" in name or "r1" in name or "qwq" in name,
        json_mode="response_format" in supported or "json" in supported,
        context_length=int(ctx),
        max_output_tokens=int(max_out),
    )


def fetch_catalogue(transport, spec, api_key: str = "") -> list[ModelInfo]:
    """Fetch and parse a provider's catalogue.

    Returns an empty list on failure rather than raising: a provider whose
    catalogue endpoint is down should leave the rest of the registry intact,
    and the caller records the failure.
    """
    headers = {"Content-Type": "application/json", **DEFAULT_HEADERS}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        raw = transport.get_json(spec.models_url or f"{spec.base_url}/models", headers)
    except Exception:
        return []
    return parse_catalogue(raw, spec)
