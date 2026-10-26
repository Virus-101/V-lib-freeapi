"""One-call API: no client to construct, no context manager to manage.

The full client exists for production use — explicit construction, dependency
injection, a mock transport in tests. Most call sites want none of that::

    import freeaipi
    freeaipi.chat("Reply with exactly: OK").content
    freeaipi.list_models()
    [c.delta for c in freeaipi.stream("Tell me a story")]

A single shared client is built on first use and reused, which matters because
construction performs network discovery. :func:`reset` rebuilds it — call it
after changing credentials so a long-lived process picks them up.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from .client import FreeAIClient
from .models import ChatChunk, ChatResponse, ModelInfo

__all__ = [
    "chat",
    "stream",
    "complete",
    "list_models",
    "no_key_models",
    "best",
    "vendors",
    "for_vendor",
    "describe",
    "run_tools",
    "status",
    "warnings",
    "reset",
    "client",
]

_client: FreeAIClient | None = None


def client() -> FreeAIClient:
    """Return the shared client, building it on first call."""
    global _client
    if _client is None:
        _client = FreeAIClient()
    return _client


def reset() -> FreeAIClient:
    """Discard the shared client and build a fresh one.

    Use after changing credentials or environment in a long-lived process;
    the cached client holds the registry from the moment it was built.
    """
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = FreeAIClient()
    return _client


# ── chat ────────────────────────────────────────────────────────────────────

def chat(messages: list[dict[str, Any]] | str, **kwargs: Any) -> ChatResponse:
    """Complete a chat turn. ``messages`` may be a plain string."""
    return client().chat(messages, **kwargs)


def complete(prompt: str, model: str = "", **kwargs: Any) -> str:
    """One-shot completion returning just the text."""
    return client().complete(prompt, model=model, **kwargs)


def stream(messages: list[dict[str, Any]] | str, **kwargs: Any) -> Iterable[ChatChunk]:
    """Stream a chat turn."""
    return client().stream(messages, **kwargs)


def run_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handlers: dict[str, Any],
    model: str = "",
    max_rounds: int = 5,
    **kwargs: Any,
) -> ChatResponse:
    """Run a tool-calling loop. See :meth:`FreeAIClient.run_tools`."""
    return client().run_tools(
        messages, tools, handlers, model=model, max_rounds=max_rounds, **kwargs
    )


# ── catalogue ───────────────────────────────────────────────────────────────

def list_models(**filters: Any) -> list[ModelInfo]:
    """List models, optionally filtered (``provider=``, ``namespace=``…)."""
    return client().list_models(**filters)


def no_key_models() -> list[ModelInfo]:
    """Models usable with no credential at all."""
    return client().no_key_models()


def for_vendor(namespace: str) -> list[ModelInfo]:
    """Every model from one vendor."""
    return client().for_vendor(namespace)


def vendors() -> list[str]:
    """Vendor namespaces currently registered."""
    return client().vendors()


def describe(key: str) -> ModelInfo:
    """Look up one model's metadata."""
    return client().describe(key)


def best(**requirements: Any) -> str:
    """Pick a model id by capability (``requires_tools=True``…)."""
    return client().best(**requirements)


# ── introspection ───────────────────────────────────────────────────────────

def status() -> dict[str, Any]:
    """Registry summary plus credential presence (never values)."""
    return client().status()


def warnings() -> list[str]:
    """Non-fatal discovery problems from the last client build."""
    return client().warnings
