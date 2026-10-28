"""freeaipi — one interface for every free LLM API.

Covers three kinds of zero-cost backend:

1. **Native free tiers** — services that serve a free model with no account
   (Pollinations) or a generous free quota.
2. **OpenRouter free models** — 17+ models on the `:free` tier, gated behind a
   free account key.
3. **Local** — Ollama and any llama.cpp / vLLM / LM Studio server on this
   machine. The only backend with no network dependency and no rate limit.

Nothing here charges money. Every model is either free, open-weight, or running
on your own hardware.
"""

from __future__ import annotations

from .client import FreeAIClient
from .convenience import (
    best,
    chat,
    complete,
    describe,
    for_vendor,
    list_models,
    no_key_models,
    reset,
    run_tools,
    status,
    stream,
    vendors,
    warnings,
)
from .models import ModelInfo, Provider, Capabilities
from .response import ChatResponse, ChatChunk, ToolCall, Usage
from .errors import (
    FreeAIPIError,
    AuthenticationError,
    RateLimitError,
    ProviderError,
    ModelNotFoundError,
    ConfigurationError,
)
from .registry import Registry, create_default_registry

__version__ = "0.2.0"

__all__ = [
    "FreeAIClient",
    "Registry",
    "create_default_registry",
    "ModelInfo",
    "Provider",
    "Capabilities",
    "ChatResponse",
    "ChatChunk",
    "ToolCall",
    "Usage",
    "FreeAIPIError",
    "AuthenticationError",
    "RateLimitError",
    "ProviderError",
    "ModelNotFoundError",
    "ConfigurationError",
    # One-call API — the common case needs no client object.
    "chat",
    "stream",
    "complete",
    "run_tools",
    "list_models",
    "no_key_models",
    "for_vendor",
    "vendors",
    "describe",
    "best",
    "status",
    "reset",
    "warnings",
    "__version__",
]
