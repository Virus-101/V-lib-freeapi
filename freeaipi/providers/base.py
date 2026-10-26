"""Provider base class.

The contract is narrow on purpose: ``chat`` for a full turn, ``stream`` for a
generator of :class:`ChatChunk`. Everything else (transport, retries, credential
resolution) is injected, so a provider is ~60 lines of vendor-specific logic.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional

from ..http import Transport
from ..models import ChatChunk, ChatResponse, ModelInfo, Provider, Usage


class BaseProvider:
    """Base class for every free backend."""

    provider: Provider = Provider.MOCK

    def __init__(self, model: ModelInfo, transport: Transport):
        self.model = model
        self.transport = transport

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: str | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: str | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Iterable[ChatChunk]:
        """Default streaming: one blocking call, one final chunk.

        Providers with real streaming override this. The fallback exists so a
        consumer written against ``stream()`` works on every provider.
        """
        resp = self.chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens,
            system=system, json_mode=json_mode, seed=seed, **kwargs,
        )
        if resp.reasoning:
            yield ChatChunk(reasoning=resp.reasoning, model_id=resp.model_id, provider=self.provider)
        yield ChatChunk(
            delta=resp.content,
            finish_reason=resp.finish_reason,
            usage=resp.usage,
            model_id=resp.model_id,
            provider=self.provider,
        )
        for tc in resp.tool_calls:
            yield ChatChunk(
                tool_call={"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                model_id=resp.model_id,
                provider=self.provider,
            )

    def health_check(self) -> bool:
        """Cheap reachability probe. False means the backend is not usable now."""
        try:
            self.chat([{"role": "user", "content": "hi"}], max_tokens=4)
            return True
        except Exception:
            return False

    def _merge_system(self, messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
        """Prepend a system message without mutating the caller's list.

        A bare string is coerced to a single user turn. The client does this,
        but providers are reachable directly through ``registry.get()``, and
        without the coercion ``list("hi")`` silently produces two one-character
        messages — a request that looks valid and returns nonsense.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        out = list(messages)
        if system:
            out.insert(0, {"role": "system", "content": system})
        return out

    @staticmethod
    def _usage_from(data: dict[str, Any]) -> Usage:
        return Usage.from_openai(data.get("usage") or {})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model.key}>"
