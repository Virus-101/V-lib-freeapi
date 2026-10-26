"""Curated specs for free-tier OpenAI-compatible endpoints.

Adding a provider is adding an entry here — a class per vendor would mean
fifteen copies of the same request-shaping code.

Each spec names the env var its key lives in. A provider is **enabled
automatically when that env var is present**, which is the behaviour that makes
"call all the free APIs" practical: set a key, the provider appears, no config
file, no code change.

Honesty about which are actually free
-------------------------------------
``free=True`` means the vendor publishes a no-cost tier. It does **not** mean
unlimited, and the ``rate_limit`` field says what the real constraint is. Three
of these are *not* free at all and are included only because they are common:
their entries are marked ``free=False`` so the catalogue cannot be read as a
promise.

Verified vs. assumed
--------------------
``verified`` records whether the endpoint was called during development. It is
not decoration: an unverified spec is a plausible guess that should be treated
as such until someone with a key exercises it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to talk to one OpenAI-compatible vendor."""

    namespace: str
    base_url: str
    key_env: str
    free: bool = True
    rate_limit: str = ""
    notes: str = ""
    verified: bool = False
    #: ``{"data": [...]}`` is standard, but some vendors deviate.
    models_url: str = ""
    #: Only register these ids; empty means "everything in the catalogue".
    models: tuple[str, ...] = ()
    #: Skip these ids (e.g. a vendor's paid-only flagship).
    exclude: tuple[str, ...] = ()
    #: Auth header style. "bearer" is universal; "query" exists for a few.
    auth_style: str = "bearer"
    extra_headers: dict[str, str] = field(default_factory=dict)


#: Known free tiers. ``verified=True`` only where the endpoint was actually
#: called during development of this library.
PRESETS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        namespace="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        free=True,
        rate_limit="Free tier: per-day request and token caps; resets daily",
        notes="Fastest inference available; Llama, Qwen, Gemma, Mistral. "
              "Free tier is rate-limited but genuinely $0.",
        verified=True,
    ),
    "cerebras": ProviderSpec(
        namespace="cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY",
        free=True,
        rate_limit="Free tier: per-hour and per-day caps",
        notes="Very high throughput on Llama and Qwen weights.",
        verified=False,
    ),
    "sambanova": ProviderSpec(
        namespace="sambanova",
        base_url="https://api.sambanova.ai/v1",
        key_env="SAMBANOVA_API_KEY",
        free=True,
        rate_limit="Free tier: per-day quota",
        notes="DeepSeek-V3, Llama-3.3, MiniMax. Catalogue endpoint is public "
              "without a key; only calling requires one.",
        verified=True,
    ),
    "together": ProviderSpec(
        namespace="together",
        base_url="https://api.together.xyz/v1",
        key_env="TOGETHER_API_KEY",
        free=True,
        rate_limit="Free tier: credit grant on signup, then paid",
        notes="Large open-weight catalogue. The free grant is a one-off, not a "
              "standing tier — treat as free-until-exhausted.",
        verified=False,
    ),
    "fireworks": ProviderSpec(
        namespace="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        key_env="FIREWORKS_API_KEY",
        free=True,
        rate_limit="Free tier: per-day token grant",
        notes="DeepSeek and Llama serving with function calling.",
        verified=False,
    ),
    "mistral": ProviderSpec(
        namespace="mistral",
        base_url="https://api.mistral.ai/v1",
        key_env="MISTRAL_API_KEY",
        free=True,
        rate_limit="'Free' (Discovery) plan: per-day message cap, small models only",
        notes="The free plan excludes the flagship models. Check the catalogue "
              "you actually get rather than assuming Mistral Large is included.",
        verified=False,
    ),
    "deepseek": ProviderSpec(
        namespace="deepseek",
        base_url="https://api.deepseek.com/v1",
        key_env="DEEPSEEK_API_KEY",
        free=False,
        rate_limit="Paid; small signup credit only",
        notes="NOT free. Included because it is a frequent question and its "
              "pricing is unusually low. Marked free=False deliberately.",
        verified=True,
    ),
    "novita": ProviderSpec(
        namespace="novita",
        base_url="https://api.novita.ai/v3/openai",
        key_env="NOVITA_API_KEY",
        free=True,
        rate_limit="Free tier: monthly token grant",
        notes="DeepSeek, Llama, Qwen and others; occasional free promotions.",
        verified=True,
    ),
    "nous": ProviderSpec(
        namespace="nous",
        base_url="https://inference-api.nousresearch.com/v1",
        key_env="NOUS_API_KEY",
        free=False,
        rate_limit="Paid; Hermes manages a rotating OAuth token",
        notes="NOT free — Hermes uses it. Shown so the endpoint is not confused "
              "with the free tiers. Marked free=False deliberately.",
        verified=True,
    ),
    "inception": ProviderSpec(
        namespace="inception",
        base_url="https://api.inceptionlabs.ai/v1",
        key_env="INCEPTION_API_KEY",
        free=False,
        rate_limit="Paid; Mercury diffusion models",
        notes="NOT free. Included for completeness.",
        verified=True,
    ),
}


def enabled_specs(env: dict[str, str] | None = None) -> list[ProviderSpec]:
    """Return presets whose key env var is present and non-empty.

    This is the auto-enable rule: a key in the environment is the user saying
    "I want this provider", so it appears without any configuration step.
    """
    env = env if env is not None else os.environ
    return [
        spec for spec in PRESETS.values()
        if env.get(spec.key_env, "").strip()
    ]


def load_config_file(path: str | Path) -> list[ProviderSpec]:
    """Load extra providers from a JSON file.

    Shape::

        {"providers": [{"namespace": "myvllm",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "key_env": "MYVLLM_KEY",     # omit for no auth
                        "free": true,
                        "models": ["qwen2.5-72b"]}]}

    ``models`` is optional — without it the catalogue is fetched. A private
    server with no ``/models`` endpoint works by listing the ids explicitly.
    """
    p = Path(path)
    if not p.is_file():
        return []
    import json

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    out: list[ProviderSpec] = []
    for entry in (raw or {}).get("providers") or []:
        if not isinstance(entry, dict) or not entry.get("base_url"):
            continue
        out.append(
            ProviderSpec(
                namespace=str(entry.get("namespace") or "custom"),
                base_url=str(entry["base_url"]),
                key_env=str(entry.get("key_env") or ""),
                free=bool(entry.get("free", True)),
                rate_limit=str(entry.get("rate_limit") or ""),
                notes=str(entry.get("notes") or ""),
                verified=False,
                models=tuple(entry.get("models") or ()),
                exclude=tuple(entry.get("exclude") or ()),
            )
        )
    return out


def config_paths(env: dict[str, str] | None = None) -> list[Path]:
    """Env vars that point at provider config files, in priority order."""
    env = env if env is not None else os.environ
    names = [env[n].strip() for n in ("FREEAIPI_PROVIDERS", "FREEAIPI_PROVIDERS_JSON") if env.get(n)]
    return [Path(n) for n in names if n]
