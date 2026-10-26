"""Pollinations Text API — the only zero-key hosted backend in the library.

Verified against the live endpoint on 2026-10-26:

* ``POST https://text.pollinations.ai/openai`` is OpenAI-shaped and works with
  **no key at all**. ``POST /openai/`` also works.
* The response carries ``user_tier: "anonymous"`` and real ``usage`` numbers.
* ``reasoning`` is returned as a separate field on the message, alongside
  ``content`` — not inside it.
* Tool calls come back in standard OpenAI ``tool_calls`` form.

The one thing that will bite a user is the anonymous quota, and it is enforced
per IP, not per key:

    HTTP 402 {"code": "FREE_LIMIT",
              "error": {"message": "Free limit reached: 1 reply every 30 seconds
                         per IP. Retry after 15s, ..."}}

A single ``referrer`` in the payload was observed to lift that limit on the
anonymous tier, so it is sent by default. A pollen key (``POLLINATIONS_API_KEY``)
is supported and takes precedence; with a key the request goes out as a bearer
token and the 402 disappears.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from .base import BaseProvider
from ..errors import RateLimitError
from ..models import Capabilities, ChatChunk, ChatResponse, ModelInfo, Provider, ToolCall

BASE_URL = "https://text.pollinations.ai"
REFERRER = "freeaipi"
DEFAULT_MODEL = "openai-fast"


class PollinationsProvider(BaseProvider):
    provider = Provider.POLLINATIONS

    def __init__(
        self,
        model: ModelInfo,
        transport,
        api_key: str = "",
        base_url: str = BASE_URL,
        referrer: str = REFERRER,
    ):
        super().__init__(model, transport)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.referrer = referrer

    # ── request shaping ─────────────────────────────────────────────────────

    def _url(self) -> str:
        # ``/openai`` is the documented OpenAI-compatible path; ``/openai/``
        # is accepted too. ``/`` alone answers with the GET-style model query,
        # which is NOT the chat endpoint.
        return f"{self.base_url}/openai"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        seed: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.provider_model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "referrer": self.referrer,
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

    # ── blocking call ───────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: str | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        started = time.time()
        msgs = self._merge_system(messages, system)
        payload = self._payload(msgs, tools, temperature, max_tokens, json_mode, seed)

        status, text = self.transport.post_raw(self._url(), payload, self._headers())

        # Pollinations reports the quota as an HTTP 402 with a JSON envelope
        # rather than raising, so it is decoded here instead of left to the
        # generic error path.
        if status == 402:
            retry_after = None
            try:
                import json

                err = json.loads(text).get("error", {})
                msg = err.get("message", "")
                import re

                m = re.search(r"retry after (\d+)", msg, re.IGNORECASE)
                retry_after = float(m.group(1)) if m else None
            except Exception:
                pass
            raise RateLimitError(
                f"Pollinations free quota reached ({self.model.key}): {text[:200]}",
                retry_after=retry_after,
            )
        if status >= 400:
            from ..errors import ProviderError

            raise ProviderError(f"Pollinations HTTP {status}: {text[:300]}", status_code=status)

        import json

        try:
            data = json.loads(text)
        except ValueError as exc:
            from ..errors import ProviderError

            raise ProviderError(f"Pollinations returned non-JSON: {text[:200]}", status_code=status) from exc

        return self._to_response(data, started)

    def _to_response(self, data: dict[str, Any], started: float) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            from ..errors import ProviderError

            raise ProviderError(f"Pollinations returned no choices: {str(data)[:200]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        # Some backends return content as a list of parts.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        tool_calls = [ToolCall.from_openai(tc) for tc in (msg.get("tool_calls") or [])]
        return ChatResponse(
            content=content,
            model_id=self.model.model_id,
            provider=self.provider,
            usage=self._usage_from(data),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or "stop",
            latency_ms=(time.time() - started) * 1000,
            reasoning=msg.get("reasoning") or "",
            raw=data,
            tier=data.get("user_tier") or "",
        )

    # ── streaming ───────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: str | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Iterable[ChatChunk]:
        msgs = self._merge_system(messages, system)
        payload = self._payload(
            msgs, tools, temperature, max_tokens, json_mode, seed, stream=True
        )
        model_id = self.model.model_id

        import json

        pending: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        for line in self.transport.stream_lines(self._url(), payload, self._headers()):
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue  # a partial line at a chunk boundary is not fatal
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue  # usage-only frames are normal
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                yield ChatChunk(delta=delta["content"], model_id=model_id, provider=self.provider)
            if delta.get("reasoning"):
                yield ChatChunk(
                    reasoning=delta["reasoning"], model_id=model_id, provider=self.provider
                )
            for tc in delta.get("tool_calls") or []:
                slot = pending.setdefault(
                    tc.get("index", 0), {"name": "", "arguments": ""}
                )
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
                            tool_call={
                                "name": slot["name"],
                                "arguments": _safe_json(slot["arguments"]),
                            },
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


# ── the model list ───────────────────────────────────────────────────────────
# ``GET /models`` currently returns exactly one anonymous-tier entry. It is read
# at runtime rather than hardcoded, so a second anonymous model appears in the
# catalog without a library change. The capability flags below are the ones that
# entry reports.

def build_models(transport) -> list[ModelInfo]:
    """Discover Pollinations' anonymous models.

    Falls back to the known entry if the endpoint is unreachable, so the
    registry is never empty on a network outage.
    """
    fallback = ModelInfo(
        model_id=DEFAULT_MODEL,
        provider=Provider.POLLINATIONS,
        provider_model_id="gpt-oss-20b",
        name="GPT-OSS 20B (Pollinations)",
        capabilities=Capabilities(
            text=True,
            vision=False,
            tools=True,
            reasoning=True,
            context_length=131072,
            max_output_tokens=8192,
        ),
        free=True,
        auth_required=False,
        rate_limit="1 reply / 30 s per IP (anonymous); unlimited with a pollen key",
        notes="No key required. Open-weight reasoning model. Referred traffic bypasses the anonymous quota.",
    )
    try:
        data = transport.get_json(f"{BASE_URL}/models")
    except Exception:
        return [fallback]

    out: list[ModelInfo] = []
    for entry in data if isinstance(data, list) else []:
        name = entry.get("name")
        if not name:
            continue
        caps = entry.get("input_modalities") or []
        aliases = entry.get("aliases") or []
        # Prefer the canonical id over an alias, but keep aliases resolvable.
        wire_id = aliases[0] if aliases else name
        out.append(
            ModelInfo(
                model_id=name,
                provider=Provider.POLLINATIONS,
                provider_model_id=wire_id,
                name=f"{name} (Pollinations)",
                capabilities=Capabilities(
                    text=True,
                    vision="image" in caps,
                    tools=bool(entry.get("tools")),
                    reasoning=bool(entry.get("reasoning")),
                    json_mode=bool(entry.get("tools")),
                    context_length=entry.get("context_length") or 131072,
                    max_output_tokens=entry.get("max_output_tokens") or 8192,
                ),
                free=True,
                auth_required=False,
                rate_limit="1 reply / 30 s per IP (anonymous); unlimited with a pollen key",
                notes=entry.get("description") or "",
            )
        )
    return out or [fallback]
