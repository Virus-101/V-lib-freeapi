"""Local backends — Ollama, llama.cpp, vLLM and LM Studio.

All four speak the OpenAI chat protocol on localhost, so they share one
implementation and differ only in default port and model-discovery path. They
are the only backends here with no network dependency, no account and no rate
limit — and the only ones that fail when a process is not running, which is why
discovery is a probe that can report "not running" instead of raising.

Ollama's own ``/api/chat`` NDJSON is supported too, because it exposes options
(such as ``num_ctx``) that the OpenAI-compat endpoint does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import BaseProvider
from ..errors import ProviderError
from ..models import ChatChunk, ChatResponse, ModelInfo, Provider, ToolCall


@dataclass(frozen=True)
class LocalTarget:
    """A discovered local server."""

    name: str
    base_url: str
    kind: str  # ollama | openai
    models: tuple[str, ...]


#: Well-known local endpoints, probed in order at discovery time.
KNOWN_TARGETS = (
    ("ollama", "http://127.0.0.1:11434", "ollama"),
    ("llamacpp", "http://127.0.0.1:8080/v1", "openai"),
    ("vllm", "http://127.0.0.1:8000/v1", "openai"),
    ("lmstudio", "http://127.0.0.1:1234/v1", "openai"),
    ("jan", "http://127.0.0.1:1337/v1", "openai"),
    ("gpt4all", "http://127.0.0.1:4891/v1", "openai"),
)


def discover(transport, extra_urls: list[str] | None = None, timeout: float = 2.0) -> list[LocalTarget]:
    """Probe every known local endpoint and return the ones answering.

    A probe that times out or refuses is skipped silently: on a normal desktop
    four of these five are always down, and that is not an error.
    """
    found: list[LocalTarget] = []
    urls = [(n, u, k) for n, u, k in KNOWN_TARGETS]
    for u in extra_urls or []:
        urls.append(("custom", u.rstrip("/"), "openai"))

    for name, url, kind in urls:
        try:
            if kind == "ollama":
                data = transport.get_json(f"{url}/api/tags", timeout=timeout)
                names = tuple(m.get("name") for m in (data or {}).get("models", []) if m.get("name"))
                if names:
                    found.append(LocalTarget(name, url, kind, names))
            else:
                data = transport.get_json(f"{url}/models", timeout=timeout)
                ids = tuple(m.get("id") for m in (data or {}).get("data", []) if m.get("id"))
                if ids:
                    found.append(LocalTarget(name, url, kind, ids))
        except Exception:
            continue
    return found


class LocalProvider(BaseProvider):
    """A local OpenAI-compatible server."""

    provider = Provider.LOCAL

    def __init__(self, model: ModelInfo, transport, base_url: str, kind: str = "openai"):
        super().__init__(model, transport)
        self.base_url = base_url.rstrip("/")
        self.kind = kind  # "ollama" or "openai"

    # ── Ollama native API ───────────────────────────────────────────────────

    def _ollama_chat(self, payload: dict[str, Any]) -> ChatResponse:
        import json
        import time

        started = time.time()
        data = self.transport.post_json(f"{self.base_url}/api/chat", payload)
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        return ChatResponse(
            content=content,
            model_id=self.model.model_id,
            provider=self.provider,
            usage=Usage_from_ollama(data.get("usage") or data),
            tool_calls=[
                ToolCall(
                    id=f"call_{i}",
                    name=(tc.get("function") or {}).get("name") or "",
                    arguments=(tc.get("function") or {}).get("arguments") or {},
                )
                for i, tc in enumerate(msg.get("tool_calls") or [])
            ],
            finish_reason=data.get("done_reason") or ("stop" if data.get("done") else ""),
            latency_ms=(time.time() - started) * 1000,
            raw=data,
            tier="local",
        )

    # ── OpenAI-compatible API ──────────────────────────────────────────────

    def _openai_chat(self, payload: dict[str, Any]) -> ChatResponse:
        import time

        started = time.time()
        data = self.transport.post_json(f"{self.base_url}/chat/completions", payload)
        if "error" in data:
            raise ProviderError(f"{self.model.key} error: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.model.key} returned no choices: {str(data)[:200]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return ChatResponse(
            content=content,
            model_id=self.model.model_id,
            provider=self.provider,
            usage=self._usage_from(data),
            tool_calls=[ToolCall.from_openai(tc) for tc in (msg.get("tool_calls") or [])],
            finish_reason=choice.get("finish_reason") or "stop",
            latency_ms=(time.time() - started) * 1000,
            raw=data,
            tier="local",
        )

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=2048, system=None,
             json_mode=False, seed=None, num_ctx: int | None = None, **kwargs: Any) -> ChatResponse:
        msgs = self._merge_system(messages, system)
        if self.kind == "ollama":
            payload: dict[str, Any] = {
                "model": self.model.provider_model_id,
                "messages": msgs,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if tools:
                payload["tools"] = tools
            if seed is not None:
                payload["options"]["seed"] = seed
            if num_ctx:
                payload["options"]["num_ctx"] = num_ctx
            if json_mode:
                payload["format"] = "json"
            return self._ollama_chat(payload)

        payload = {
            "model": self.model.provider_model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        if num_ctx:
            payload["max_context_len"] = num_ctx
        return self._openai_chat(payload)

    def stream(self, messages, tools=None, temperature=0.7, max_tokens=2048, system=None,
               json_mode=False, seed=None, num_ctx=None, **kwargs: Any) -> Iterable[ChatChunk]:
        msgs = self._merge_system(messages, system)
        model_id = self.model.model_id

        if self.kind == "ollama":
            payload = {
                "model": self.model.provider_model_id,
                "messages": msgs,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if tools:
                payload["tools"] = tools
            if num_ctx:
                payload["options"]["num_ctx"] = num_ctx
            import json

            for line in self.transport.stream_lines(f"{self.base_url}/api/chat", payload):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                msg = obj.get("message") or {}
                if msg.get("content"):
                    yield ChatChunk(delta=msg["content"], model_id=model_id, provider=self.provider)
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    yield ChatChunk(
                        tool_call={"name": fn.get("name") or "", "arguments": fn.get("arguments") or {}},
                        model_id=model_id,
                        provider=self.provider,
                    )
                if obj.get("done"):
                    yield ChatChunk(
                        finish_reason=obj.get("done_reason") or "stop",
                        usage=Usage_from_ollama(obj),
                        model_id=model_id,
                        provider=self.provider,
                    )
            return

        payload = {
            "model": self.model.provider_model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if num_ctx:
            payload["max_context_len"] = num_ctx
        yield from self._stream_openai(payload, model_id)

    def _stream_openai(self, payload: dict[str, Any], model_id: str) -> Iterable[ChatChunk]:
        import json

        pending: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        for line in self.transport.stream_lines(
            f"{self.base_url}/chat/completions", payload
        ):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                yield ChatChunk(delta=delta["content"], model_id=model_id, provider=self.provider)
            for tc in delta.get("tool_calls") or []:
                slot = pending.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            finish = choices[0].get("finish_reason")
            if finish:
                for slot in pending.values():
                    if slot["name"]:
                        yield ChatChunk(
                            tool_call={"name": slot["name"], "arguments": _safe_json(slot["arguments"])},
                            finish_reason=finish,
                            model_id=model_id,
                            provider=self.provider,
                        )
                pending.clear()
                yield ChatChunk(
                    finish_reason=finish,
                    usage=self._usage_from({"usage": usage}),
                    model_id=model_id,
                    provider=self.provider,
                )

    def health_check(self) -> bool:
        try:
            if self.kind == "ollama":
                self.transport.get_json(f"{self.base_url}/api/tags", timeout=5.0)
            else:
                self.transport.get_json(f"{self.base_url}/models", timeout=5.0)
            return True
        except Exception:
            return False


def Usage_from_ollama(data: dict[str, Any]):
    """Map Ollama's token counters onto the shared Usage shape.

    Ollama reports evaluation counts rather than the OpenAI triple, so the
    prompt/completion split is preserved where it exists and total is left at 0
    when the backend did not send one.
    """
    from ..models import Usage

    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    if prompt is None and completion is None:
        return Usage()
    return Usage(
        prompt_tokens=int(prompt or 0),
        completion_tokens=int(completion or 0),
        total_tokens=int((prompt or 0) + (completion or 0)),
    )


def _safe_json(text: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(text or "{}")
        return parsed if isinstance(parsed, dict) else {"_raw": text}
    except (ValueError, TypeError):
        return {"_raw": text}


def build_models(transport, base_url: str, kind: str, names: Iterable[str]) -> list[ModelInfo]:
    """Turn discovered local model names into registry entries."""
    out: list[ModelInfo] = []
    for name in names:
        bare = name.split(":")[0]
        out.append(
            ModelInfo(
                model_id=f"{bare}",
                provider=Provider.LOCAL,
                provider_model_id=name,
                name=f"{bare} (local)",
                capabilities=_infer_caps(bare),
                free=True,
                auth_required=False,
                rate_limit="",
                notes=f"{kind} @ {base_url}",
            )
        )
    return out


_SIZE_CTX = (
    (100, 1_000_000), (512, 512_000), (256, 262_144), (128, 131_072),
    (64, 65_536), (32, 32_768), (16, 16_384), (8, 8_192), (4, 4_096),
)


def _infer_caps(name: str) -> Any:
    """Guess context length and tool support from a model name.

    A local model's metadata is whatever the GGUF header says, and reading it
    means parsing the file. Name-based inference is wrong for some models and
    right for most; the value is a hint, not a contract, and is marked as such
    in the notes field.
    """
    from ..models import Capabilities

    lowered = name.lower()
    ctx = 8192
    for marker, size in _SIZE_CTX:
        if f"{marker}k" in lowered or f"{marker}k" in lowered.replace("_", ""):
            ctx = size
            break
    # Vision flags appear anywhere in the name: "llava-13b", "qwen2.5-vl-7b",
    # "model-vl". Checking only the last dash segment misses "vl" in the middle.
    segments = lowered.replace(":", "-").split("-")
    vision = "vision" in lowered or "llava" in lowered or "vl" in segments
    tools = any(k in lowered for k in ("tool", "function", "agentic", "coder", "instruct"))
    reasoning = any(k in lowered for k in ("reason", "think", "r1", "qwq", "deepseek"))
    return Capabilities(
        text=True,
        vision=vision,
        tools=tools,
        reasoning=reasoning,
        context_length=ctx,
        max_output_tokens=min(8192, ctx // 2),
    )
