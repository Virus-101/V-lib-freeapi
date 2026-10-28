"""Per-developer local state: keys and a usage ledger.

Every developer runs on their own free-tier accounts, so nothing here is shared
or sent anywhere. State lives in one directory (``~/.freeaipi`` by default,
``FREEAIPI_HOME`` to override):

* ``keys.env``    — ``NAME=value`` lines, one per provider key, mode 0600.
* ``usage.jsonl`` — one line per completed call: when, model, token counts.

Keys loaded from ``keys.env`` never override an env var that is already set, so
a CI job or a one-off ``GROQ_API_KEY=... python app.py`` still wins.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .presets import PRESETS

#: Where to get a free key. Only vendors whose free tier needs an account.
SIGNUP: dict[str, str] = {
    "openrouter": "https://openrouter.ai/keys",
    "groq": "https://console.groq.com/keys",
    "cerebras": "https://cloud.cerebras.ai",
    "sambanova": "https://cloud.sambanova.ai",
    "mistral": "https://console.mistral.ai/api-keys",
    "together": "https://api.together.ai/settings/api-keys",
    "fireworks": "https://fireworks.ai/account/api-keys",
    "novita": "https://novita.ai/settings/key-management",
}


def home() -> Path:
    """The state directory. Not created until something is written."""
    env = os.environ.get("FREEAIPI_HOME", "").strip()
    return Path(env) if env else Path.home() / ".freeaipi"


def keys_path() -> Path:
    return home() / "keys.env"


def usage_path() -> Path:
    return home() / "usage.jsonl"


def key_names() -> dict[str, str]:
    """``{vendor: ENV_VAR}`` for every vendor whose free tier takes a key."""
    out = {"openrouter": "OPENROUTER_API_KEY"}
    for ns, spec in PRESETS.items():
        if spec.free and spec.key_env:
            out[ns] = spec.key_env
    return out


def read_keys() -> dict[str, str]:
    """Parse ``keys.env``. Missing or unreadable file means no keys."""
    path = keys_path()
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if name.strip() and value:
                out[name.strip()] = value
    except OSError:
        return {}
    return out


def load_keys() -> list[str]:
    """Export stored keys into ``os.environ`` without overriding existing ones.

    Returns the env var names that were set, never their values.
    """
    applied = []
    for name, value in read_keys().items():
        if not os.environ.get(name, "").strip():
            os.environ[name] = value
            applied.append(name)
    return applied


def set_key(env_var: str, value: str) -> Path:
    """Store one key, replacing any earlier value for the same variable."""
    keys = read_keys()
    keys[env_var] = value.strip()
    path = keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "# freeaipi keys: private to this machine, do not commit\n"
    body += "".join(f"{k}={v}\n" for k, v in sorted(keys.items()))
    path.write_text(body, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # Windows: ACLs, not mode bits
        pass
    return path


def remove_key(env_var: str) -> bool:
    keys = read_keys()
    if env_var not in keys:
        return False
    del keys[env_var]
    path = keys_path()
    body = "# freeaipi keys: private to this machine, do not commit\n"
    body += "".join(f"{k}={v}\n" for k, v in sorted(keys.items()))
    path.write_text(body, encoding="utf-8")
    return True


# ── usage ledger ────────────────────────────────────────────────────────────

def record_usage(model_key: str, provider: str, prompt: int, completion: int, total: int) -> None:
    """Append one call to the ledger. Failure to write is never an error."""
    try:
        path = usage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {"t": int(time.time()), "model": model_key, "provider": provider,
                "prompt": prompt, "completion": completion, "total": total}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError:
        pass


def usage_summary(since: float = 0.0) -> dict[str, Any]:
    """Aggregate the ledger: per-model and overall token and call counts."""
    path = usage_path()
    per_model: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "tokens": 0})
    calls = tokens = 0
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get("t", 0) < since:
                continue
            n = int(row.get("total") or 0)
            per_model[row.get("model", "?")]["calls"] += 1
            per_model[row.get("model", "?")]["tokens"] += n
            calls += 1
            tokens += n
    return {"calls": calls, "tokens": tokens, "models": dict(per_model)}
