"""Data types shared by every provider.

Deliberately small. A provider converts its own wire format into these, so the
client and the registry never learn what a given vendor calls its fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    """A zero-cost backend.

    Values are the ids used in the registry key ``<provider>/<model>`` and in
    logs. ``LOCAL`` is a family, not a host: Ollama, llama.cpp, vLLM and
    LM Studio all register under it.
    """

    POLLINATIONS = "pollinations"
    OPENROUTER = "openrouter"
    INCEPTION = "inception"
    LOCAL = "local"
    MOCK = "mock"
    #: Any OpenAI-compatible endpoint configured at runtime (Groq, Cerebras,
    #: SambaNova, Together, Fireworks, Mistral, DeepSeek, Novita, a private
    #: vLLM…). The *kind* is always this member; the per-vendor name lives in
    #: :attr:`ModelInfo.namespace`, so ``Provider.COMPAT`` stays a stable
    #: dispatch target while ``groq/llama-3.3`` reads correctly in the registry.
    COMPAT = "openai_compat"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class Capabilities:
    """What a model can do. Defaults are the conservative case."""

    text: bool = True
    vision: bool = False
    tools: bool = False
    reasoning: bool = False
    json_mode: bool = False
    context_length: int = 8192
    max_output_tokens: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "vision": self.vision,
            "tools": self.tools,
            "reasoning": self.reasoning,
            "json_mode": self.json_mode,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass
class ModelInfo:
    """A model entry in the registry.

    ``provider_model_id`` is the id the backend expects on the wire; it differs
    from ``model_id`` when a provider is addressed through an alias
    (Pollinations serves ``gpt-oss-20b`` under the alias ``openai-fast``).

    ``namespace`` is the name used in the registry key and in logs. It defaults
    to the provider kind and is overridden for generic OpenAI-compatible
    endpoints so the key reads ``groq/llama-3.3-70b`` rather than
    ``openai_compat/llama-3.3-70b`` — the kind is still ``Provider.COMPAT`` for
    dispatch, but the *vendor* is what a user types.
    """

    model_id: str
    provider: Provider
    provider_model_id: str
    name: str
    capabilities: Capabilities = field(default_factory=Capabilities)
    free: bool = True
    auth_required: bool = False
    rate_limit: str = ""
    notes: str = ""
    namespace: str | None = None

    @property
    def label(self) -> str:
        """The vendor name used in the registry key."""
        return self.namespace or self.provider.value

    @property
    def key(self) -> str:
        """Registry key, ``<namespace>/<model_id>``."""
        return f"{self.label}/{self.model_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "namespace": self.label,
            "provider_model_id": self.provider_model_id,
            "name": self.name,
            "key": self.key,
            "free": self.free,
            "auth_required": self.auth_required,
            "rate_limit": self.rate_limit,
            "notes": self.notes,
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> "ToolCall":
        """Build from an OpenAI-style tool_call object.

        The arguments arrive as a JSON *string* on the wire. A malformed string
        must not raise — it is preserved under ``_raw`` so the caller sees the
        failure instead of losing the turn.
        """
        fn = raw.get("function") or {}
        args_raw = fn.get("arguments", "")
        args: dict[str, Any]
        if isinstance(args_raw, dict):
            args = args_raw
        else:
            import json

            try:
                parsed = json.loads(args_raw or "{}")
                args = parsed if isinstance(parsed, dict) else {"_raw": args_raw}
            except (ValueError, TypeError):
                args = {"_raw": args_raw}
        return cls(id=raw.get("id") or "", name=fn.get("name") or "", arguments=args)


@dataclass
class Usage:
    """Token accounting, when the provider reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, raw: dict[str, Any] | None) -> "Usage":
        raw = raw or {}
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )

    @property
    def cost_usd(self) -> float:
        """Always 0.0 — every model in this library is free."""
        return 0.0


@dataclass
class ChatResponse:
    """A completed chat turn."""

    content: str
    model_id: str
    provider: Provider
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    tier: str = ""

    @property
    def cost_usd(self) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model_id": self.model_id,
            "provider": self.provider.value,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost_usd": self.cost_usd,
            },
            "tool_calls": [
                {"id": t.id, "name": t.name, "arguments": t.arguments} for t in self.tool_calls
            ],
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 1),
            "reasoning": self.reasoning,
            "tier": self.tier,
        }


@dataclass
class ChatChunk:
    """One streamed fragment.

    Exactly one of ``delta`` / ``reasoning`` / ``tool_call`` / ``finish_reason``
    is set. This is the contract the streaming consumer dispatches on — a chunk
    that set two fields would make the caller double-count text.
    """

    delta: str = ""
    reasoning: str = ""
    tool_call: dict[str, Any] | None = None
    finish_reason: str = ""
    usage: Usage | None = None
    model_id: str = ""
    provider: Provider | None = None
