"""The public client.

One object holds credentials, the registry and the transport, and offers
``chat`` / ``stream`` / ``complete``. Routing is explicit by default (you name
the model) with an optional capability-based selector for "give me any free
model that can do X".
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from .credentials import resolve_key
from .errors import ConfigurationError, ModelNotFoundError, RateLimitError
from .http import Transport
from .models import Capabilities, ChatChunk, ChatResponse, ModelInfo, Provider, ToolCall
from .registry import Registry, create_default_registry


@dataclass
class ChatOptions:
    """Everything a chat call can tune. Defaults match the free tiers."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    system: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: list[dict[str, Any]] | None = None
    json_mode: bool = False
    seed: int | None = None
    timeout: float | None = None
    fallback: list[str] = field(default_factory=list)
    on_retry: Callable[[str, float], None] | None = None


class FreeAIClient:
    """Client for every free LLM backend.

    Constructing it performs discovery, so it is not free: build one and reuse
    it. ``openrouter_key`` and ``pollinations_key`` may be passed explicitly or
    read from ``OPENROUTER_API_KEY`` / ``POLLINATIONS_API_KEY``.
    """

    def __init__(
        self,
        openrouter_key: str | None = None,
        pollinations_key: str | None = None,
        local_urls: list[str] | None = None,
        transport: Transport | None = None,
        include_local: bool = True,
        include_openrouter: bool = True,
        include_pollinations: bool = True,
        max_retries: int = 3,
    ):
        self._or_key = resolve_key(openrouter_key, "openrouter")
        self._pl_key = resolve_key(pollinations_key, "pollinations")
        self.transport = transport or Transport(max_retries=max_retries)

        self.registry = create_default_registry(
            transport=self.transport,
            openrouter_key=self._or_key.value,
            pollinations_key=self._pl_key.value,
            local_urls=local_urls,
            include_local=include_local,
            include_openrouter=include_openrouter,
            include_pollinations=include_pollinations,
        )

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "FreeAIClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def warnings(self) -> list[str]:
        """Non-fatal discovery problems. Empty means every backend answered."""
        return self.registry.warnings

    # ── catalogue ───────────────────────────────────────────────────────────

    def list_models(self, **filters: Any) -> list[ModelInfo]:
        """List models, optionally filtered (``provider=``, ``requires_tools=``…)."""
        return self.registry.list(**filters)

    def no_key_models(self) -> list[ModelInfo]:
        """Models usable with no credential at all."""
        return self.registry.list(auth_required=False)

    def describe(self, key: str) -> ModelInfo:
        model, _ = self.registry.get(key)
        return model

    # ── chat ────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]] | str,
        **kwargs: Any,
    ) -> ChatResponse:
        """Complete a chat turn.

        ``messages`` may be a plain string (converted to a single user turn) or
        an OpenAI-style list. Returns when the full reply is ready.
        """
        opts = self._opts(messages, kwargs)
        targets = self._targets(opts)
        if not targets:
            raise ModelNotFoundError(
                "No model matched. Registry has: "
                f"{', '.join(sorted(self.registry.models)) or '(none)'}"
            )

        last_exc: Exception | None = None
        for key in targets:
            model, provider = self.registry.get(key)
            try:
                return provider.chat(
                    messages=opts.messages,
                    tools=opts.tools,
                    temperature=opts.temperature,
                    max_tokens=opts.max_tokens,
                    system=opts.system,
                    json_mode=opts.json_mode,
                    seed=opts.seed,
                )
            except RateLimitError:
                raise  # quota is not a reason to silently switch models
            except Exception as exc:
                last_exc = exc
                if key != targets[-1]:
                    continue
                raise
        assert last_exc is not None  # unreachable: targets is non-empty
        raise last_exc

    def stream(
        self,
        messages: list[dict[str, Any]] | str,
        **kwargs: Any,
    ) -> Iterable[ChatChunk]:
        """Stream a chat turn. Same arguments as :meth:`chat`."""
        opts = self._opts(messages, kwargs)
        targets = self._targets(opts)
        if not targets:
            raise ModelNotFoundError(
                "No model matched. Registry has: "
                f"{', '.join(sorted(self.registry.models)) or '(none)'}"
            )
        key = targets[0]
        model, provider = self.registry.get(key)
        return provider.stream(
            messages=opts.messages,
            tools=opts.tools,
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            system=opts.system,
            json_mode=opts.json_mode,
            seed=opts.seed,
        )

    def complete(
        self, prompt: str, model: str = "", system: str | None = None, **kwargs: Any
    ) -> str:
        """Convenience: a one-shot completion returning just the text."""
        resp = self.chat(prompt, model=model, system=system, **kwargs)
        return resp.content

    # ── tool loop ───────────────────────────────────────────────────────────

    def run_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        handlers: dict[str, Callable[..., Any]],
        model: str = "",
        max_rounds: int = 5,
        **kwargs: Any,
    ) -> ChatResponse:
        """Run a tool-calling loop until the model stops calling tools.

        Each round: ask, execute every requested tool, append the results as
        ``tool`` messages, repeat. Stops when the model replies without tool
        calls, or after ``max_rounds`` — the round count is reported on the
        returned response's ``raw['rounds']`` either way, so a silent stop at
        the cap is visible rather than looking like a natural finish.
        """
        msgs = list(messages)
        for round_no in range(1, max_rounds + 1):
            resp = self.chat(msgs, model=model, tools=tools, **kwargs)
            if not resp.tool_calls:
                resp.raw["rounds"] = round_no
                return resp

            msgs.append(
                {
                    "role": "assistant",
                    "content": resp.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )
            for tc in resp.tool_calls:
                handler = handlers.get(tc.name)
                try:
                    result = handler(**tc.arguments) if handler else f"no handler for {tc.name}"
                except Exception as exc:
                    result = f"error: {exc}"
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        # Cap reached: one final call with tools removed so the model summarises.
        resp = self.chat(msgs, model=model, **kwargs)
        resp.raw["rounds"] = max_rounds
        resp.raw["round_cap_reached"] = True
        return resp

    # ── routing ─────────────────────────────────────────────────────────────

    def best(
        self,
        requires_tools: bool | None = None,
        requires_vision: bool | None = None,
        requires_reasoning: bool | None = None,
        prefer_no_key: bool = True,
    ) -> str:
        """Pick a model id by capability, preferring one that needs no key.

        Ranking: no-key first, then by context window. Explicitly deterministic
        — given the same registry it returns the same id, which matters for
        caching and for tests.
        """
        candidates = self.registry.list(
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_reasoning=requires_reasoning,
        )
        if not candidates:
            raise ModelNotFoundError(
                "No model matches those requirements. Try: "
                + ", ".join(m.key for m in self.registry.list()[:5])
            )
        candidates.sort(key=lambda m: (m.auth_required if prefer_no_key else False, -m.capabilities.context_length))
        return candidates[0].key

    def _opts(self, messages: list[dict[str, Any]] | str, kwargs: dict[str, Any]) -> ChatOptions:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        known = {
            "model", "system", "temperature", "max_tokens", "tools",
            "json_mode", "seed", "timeout", "fallback", "on_retry",
        }
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            raise TypeError(f"Unknown chat option(s): {', '.join(sorted(extra))}")
        return ChatOptions(messages=messages, **kwargs)  # type: ignore[arg-type]

    def _targets(self, opts: ChatOptions) -> list[str]:
        """Resolve the ordered list of model keys to try."""
        if opts.model:
            return [opts.model] + list(opts.fallback)
        if opts.fallback:
            return list(opts.fallback)
        # Only filter on tools when tools were actually supplied. Passing
        # ``requires_tools=False`` for a plain text request would filter out
        # every tool-capable model and leave only models that cannot call tools.
        requires_tools = True if opts.tools else None
        return [self.best(requires_tools=requires_tools)]

    # ── introspection ───────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Registry summary plus credential presence (never values)."""
        from .credentials import _redact

        return {
            "models": len(self.registry),
            "providers": self.registry.providers_in_use(),
            "no_key": [m.key for m in self.registry.list(auth_required=False)],
            "needs_key": [m.key for m in self.registry.list(auth_required=True)],
            "credentials": {
                "openrouter": _redact(self._or_key.value) if self._or_key.present else "<absent>",
                "pollinations": _redact(self._pl_key.value) if self._pl_key.present else "<absent>",
            },
            "warnings": self.registry.warnings,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FreeAIClient models={len(self.registry)} providers={self.registry.providers_in_use()}>"
