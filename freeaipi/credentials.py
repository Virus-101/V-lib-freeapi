"""Credentials resolved per request, never captured at construction.

Hosted free tiers publish hard quotas — Pollinations allows one anonymous reply
per 30 seconds per IP. A key or token read once at startup is a bug waiting to
happen, so every credential here is a callable invoked at request time.

Resolution order for a provider key: an explicit value passed to the client,
then the env var, then any override file. A durable user key always wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import ConfigurationError

#: Env var consulted for each provider. Documented in README.
ENV_VARS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "inception": "INCEPTION_API_KEY",
}


def _redact(key: str) -> str:
    """Render a credential for logs. Never the value itself."""
    if not key:
        return "<absent>"
    if len(key) <= 8:
        return f"<set len={len(key)}>"
    return f"<set len={len(key)} prefix={key[:4]}…>"


@dataclass(frozen=True)
class Credential:
    """A resolved credential plus where it came from.

    ``source`` is one of ``explicit``, ``env``, ``file``, ``none``. It exists so
    a caller can report "your key was read from OPENROUTER_API_KEY" without ever
    printing the key.
    """

    value: str = ""
    source: str = "none"

    @property
    def present(self) -> bool:
        return bool(self.value)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return _redact(self.value)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Credential(source={self.source!r}, value={_redact(self.value)})"


def resolve_key(
    explicit: str | None,
    provider: str,
    env_var: str | None = None,
    key_file: Path | None = None,
) -> Credential:
    """Resolve a provider key.

    Order: ``explicit`` → env var → ``key_file`` → absent. An empty string is
    treated as absent, because ``os.environ.get(...)`` returning ``""`` is how a
    half-configured shell presents itself.
    """
    if explicit:
        return Credential(explicit, "explicit")

    var = env_var or ENV_VARS.get(provider)
    if var:
        from_env = os.environ.get(var, "").strip()
        if from_env:
            return Credential(from_env, f"env:{var}")

    if key_file is not None and Path(key_file).is_file():
        text = Path(key_file).read_text(encoding="utf-8").strip()
        if text:
            return Credential(text.splitlines()[0].strip(), f"file:{key_file}")

    return Credential("", "none")


def require_key(cred: Credential, provider: str, hint: str = "") -> str:
    """Return the key or raise :class:`ConfigurationError`.

    The message names the env var and the file, never the value.
    """
    if not cred.present:
        raise ConfigurationError(
            f"{provider} requires an API key (found: {cred.source}). "
            f"Set {ENV_VARS.get(provider, provider.upper() + '_API_KEY')}"
            + (f", or pass {hint}." if hint else ".")
        )
    return cred.value


KeySource = Callable[[], str]
