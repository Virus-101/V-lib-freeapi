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

from .errors import ConfigurationError, ModelNotFoundError
from .http import Transport
from .models import Capabilities, ModelInfo, Provider
from .providers.base import BaseProvider
from .providers.local import LocalProvider, build_models as local_build_models, discover
from .providers.openrouter import OpenRouterProvider, build_models as or_build_models
from .providers.pollinations import PollinationsProvider, build_models as pl_build_models


def _coerce_provider(value: Provider | str) -> Provider:
    """Accept ``Provider.LOCAL`` or ``"local"``; raise on anything unknown."""
    if isinstance(value, Provider):
        return value
    try:
        return Provider(value)
    except ValueError:
        raise ModelNotFoundError(
            f"Unknown provider kind '{value}'. Known: "
            + ", ".join(p.value for p in Provider)
        ) from None


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
        namespace: str | None = None,
    ) -> list[ModelInfo]:
        """Filter the catalogue.

        ``provider`` accepts a :class:`Provider` or a provider-kind string.
        ``namespace`` filters on the *vendor* name (``groq``, ``cerebras``),
        which is what you want for generic OpenAI-compatible endpoints where
        many vendors share one provider kind. ``None`` means "do not filter".
        """
        out = []
        for m in self.models.values():
            if prefix and not m.key.startswith(prefix):
                continue
            if namespace is not None and m.label != namespace:
                continue
            if provider is not None and m.provider != _coerce_provider(provider):
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
        """Group by vendor namespace (``groq``, ``pollinations``, ``local``)."""
        out: dict[str, list[ModelInfo]] = {}
        for m in self.models.values():
            out.setdefault(m.label, []).append(m)
        return out

    def providers_in_use(self) -> list[str]:
        return sorted({m.label for m in self.models.values()})

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
    include_compat: bool = True,
    compat_specs: list[Any] | None = None,
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

    # ── Generic OpenAI-compatible free tiers ──────────────────────────────
    if include_compat:
        from .presets import enabled_specs, load_config_file, config_paths
        from .providers.openai_compat import OpenAICompatProvider, fetch_catalogue

        specs = list(compat_specs) if compat_specs is not None else enabled_specs()
        for path in config_paths():
            try:
                specs.extend(load_config_file(path))
            except Exception as exc:
                reg.warnings.append(f"provider config {path} failed to load: {exc}")

        for spec in specs:
            try:
                # Inside the try on purpose: a provider whose key is missing or
                # whose catalogue is unreachable must leave the rest of the
                # registry intact, exactly like the other discovery steps.
                key = _spec_key(spec)
                entries = fetch_catalogue(transport, spec, key)
                if spec.models:
                    entries = [m for m in entries if m.model_id in spec.models] or [
                        _spec_only_model(spec, mid) for mid in spec.models
                    ]
                if spec.exclude:
                    entries = [m for m in entries if m.model_id not in spec.exclude]
                if not entries:
                    reg.warnings.append(
                        f"{spec.namespace}: no models discovered (key present but "
                        "catalogue empty or unreachable)"
                    )
                for m in entries:
                    reg.register(m, OpenAICompatProvider(m, transport, spec, spec.base_url, key))
            except Exception as exc:
                reg.warnings.append(f"{spec.namespace} discovery failed: {exc}")

    return reg


def _spec_key(spec) -> str:
    """Resolve a preset's credential from its env var.

    Raises ``ConfigurationError`` only if the spec was handed in directly with
    no key; auto-enabled specs already have one by construction.
    """
    from .credentials import resolve_key

    cred = resolve_key(None, spec.namespace, env_var=spec.key_env or None)
    if not cred.present and not spec.key_env:
        # A config-file provider with no key_env and no key is a public
        # endpoint (e.g. a local vLLM) — that is legitimate.
        return ""
    if not cred.present:
        raise ConfigurationError(
            f"{spec.namespace} is enabled but {spec.key_env} is not set."
        )
    return cred.value


def _spec_only_model(spec, model_id: str) -> Any:
    """Build a registry entry for an id named in config with no catalogue."""
    return ModelInfo(
        model_id=model_id,
        provider=Provider.COMPAT,
        provider_model_id=model_id,
        name=model_id,
        capabilities=Caps_from_id(model_id),
        free=spec.free,
        auth_required=bool(spec.key_env),
        rate_limit=spec.rate_limit,
        notes=spec.notes,
        namespace=spec.namespace,
    )


def Caps_from_id(model_id: str):
    """Name-based capability guess for a model named in config only."""
    from .providers.local import _infer_caps

    return _infer_caps(model_id)
