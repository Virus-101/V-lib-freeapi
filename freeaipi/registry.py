"""The registry: which models exist, and which provider instance serves them.

The registry is built at client construction and is immutable afterwards. It
merges three sources:

1. Static entries that are always true (local models, if a server is up).
2. Pollinations' ``/models`` — one anonymous entry today.
3. OpenRouter's ``:free`` tier — fetched only when a key is present.

Discovery failures are non-fatal and recorded in ``Registry.warnings``: a
network outage must not make the library unusable, but it must be visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .errors import ModelNotFoundError
from .http import Transport
from .models import Capabilities, ModelInfo, Provider
from .providers.base import BaseProvider
from .providers.local import LocalProvider, build_models as local_build_models, discover
from .providers.openrouter import OpenRouterProvider, build_models as or_build_models
from .providers.pollinations import PollinationsProvider, build_models as pl_build_models


@dataclass
class Registry:
    """Model catalogue plus the provider instance behind each entry."""

    models: dict[str, ModelInfo] = field(default_factory=dict)
    providers: dict[str, BaseProvider] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ── lookups ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> tuple[ModelInfo, BaseProvider]:
        """Return ``(model_info, provider)`` for a registry key.

        Accepts ``provider/model``, bare ``model``, and ``provider/model`` with
        the provider omitted when the model id is unique. Ambiguity raises rather
        than guessing.
        """
        if key in self.models:
            return self.models[key], self.providers[key]

        # bare model id — unique match required
        matches = [k for k in self.models if k.split("/", 1)[-1] == key]
        if len(matches) == 1:
            return self.models[matches[0]], self.providers[matches[0]]
        if len(matches) > 1:
            raise ModelNotFoundError(
                f"'{key}' is ambiguous; use one of: {', '.join(sorted(matches))}"
            )
        raise ModelNotFoundError(
            f"Model '{key}' not in registry. Known: {', '.join(sorted(self.models)) or '(none)'}"
        )

    def __contains__(self, key: str) -> bool:
        return key in self.models

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self.models.values())

    def __len__(self) -> int:
        return len(self.models)

    # ── queries ─────────────────────────────────────────────────────────────

    def list(
        self,
        provider: Provider | str | None = None,
        requires_tools: bool | None = None,
        requires_vision: bool | None = None,
        requires_reasoning: bool | None = None,
        auth_required: bool | None = None,
        prefix: str = "",
    ) -> list[ModelInfo]:
        """Filter the catalogue. ``None`` means "do not filter on this"."""
        if isinstance(provider, str):
            provider = Provider(provider)
        out = []
        for m in self.models.values():
            if prefix and not m.key.startswith(prefix):
                continue
            if provider is not None and m.provider != provider:
                continue
            c = m.capabilities
            if requires_tools is not None and c.tools != requires_tools:
                continue
            if requires_vision is not None and c.vision != requires_vision:
                continue
            if requires_reasoning is not None and c.reasoning != requires_reasoning:
                continue
            if auth_required is not None and m.auth_required != auth_required:
                continue
            out.append(m)
        return out

    def by_provider(self) -> dict[str, list[ModelInfo]]:
        out: dict[str, list[ModelInfo]] = {}
        for m in self.models.values():
            out.setdefault(m.provider.value, []).append(m)
        return out

    def providers_in_use(self) -> list[str]:
        return sorted({m.provider.value for m in self.models.values()})

    # ── mutation ────────────────────────────────────────────────────────────

    def register(self, model: ModelInfo, provider: BaseProvider) -> None:
        self.models[model.key] = model
        self.providers[model.key] = provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.models),
            "providers": self.providers_in_use(),
            "models": [m.to_dict() for m in self.models.values()],
            "warnings": list(self.warnings),
        }


def create_default_registry(
    transport: Transport | None = None,
    openrouter_key: str = "",
    pollinations_key: str = "",
    local_urls: list[str] | None = None,
    include_local: bool = True,
    include_openrouter: bool = True,
    include_pollinations: bool = True,
) -> Registry:
    """Build a registry with discovery.

    Every discovery step is wrapped individually: one failing backend leaves
    the others populated, and the failure is recorded in ``warnings``.
    """
    transport = transport or Transport()
    reg = Registry()

    # ── Pollinations (no key) ─────────────────────────────────────────────
    if include_pollinations:
        try:
            for m in pl_build_models(transport):
                reg.register(
                    m, PollinationsProvider(m, transport, api_key=pollinations_key)
                )
        except Exception as exc:
            reg.warnings.append(f"pollinations discovery failed: {exc}")

    # ── OpenRouter (key required for chat, not for the catalogue) ────────
    if include_openrouter:
        try:
            catalogue = or_build_models(transport, api_key=openrouter_key)
        except Exception as exc:
            catalogue = []
            reg.warnings.append(f"openrouter discovery failed: {exc}")

        if catalogue and not openrouter_key:
            # One line, not one per model: 17 identical messages read as a
            # broken library rather than a missing key.
            reg.warnings.append(
                f"openrouter: {len(catalogue)} free models known but unavailable — "
                "set OPENROUTER_API_KEY to use them"
            )
            catalogue = []
        else:
            for m in catalogue:
                try:
                    reg.register(m, OpenRouterProvider(m, transport, api_key=openrouter_key))
                except Exception as exc:
                    reg.warnings.append(f"openrouter model {m.key} unavailable: {exc}")

    # ── Local ─────────────────────────────────────────────────────────────
    if include_local:
        try:
            for target in discover(transport, extra_urls=local_urls):
                for m in local_build_models(transport, target.base_url, target.kind, target.models):
                    reg.register(
                        m, LocalProvider(m, transport, target.base_url, kind=target.kind)
                    )
        except Exception as exc:
            reg.warnings.append(f"local discovery failed: {exc}")

    return reg
