"""A local OpenAI-compatible gateway over the free backends.

``fllm serve`` starts it; any tool that speaks the OpenAI API (an IDE plugin,
LangChain, the ``openai`` SDK) points at ``http://127.0.0.1:8787/v1`` and gets
free models behind one address. It binds to loopback only and uses the
developer's own keys, so there is nothing to host and nothing shared.

Standard library only, so it adds no dependencies.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .client import FreeAIClient
from .errors import FreeAIPIError, RateLimitError

_PASS = ("temperature", "max_tokens", "tools", "seed")


def _make_handler(client: FreeAIClient):
    class Handler(BaseHTTPRequestHandler):
        server_version = "freeaipi-gateway"

        def log_message(self, *args: Any) -> None:  # keep stdout quiet
            pass

        def _json(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str, kind: str) -> None:
            self._json(code, {"error": {"message": message, "type": kind}})

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                models = [m for m in client.list_models()]
                self._json(200, {"object": "list", "data": [
                    {"id": m.key, "object": "model", "owned_by": m.label} for m in models]})
            elif self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"ok": True, "models": len(client.registry)})
            else:
                self._error(404, "not found", "not_found")

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._error(404, "not found", "not_found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                messages = body["messages"]
            except (ValueError, KeyError):
                return self._error(400, "body must be JSON with a messages list", "invalid_request")
            kwargs = {k: body[k] for k in _PASS if body.get(k) is not None}
            model = body.get("model") or ""
            if model in ("", "auto", "free"):
                model = ""
            if model:
                kwargs["model"] = model
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            try:
                if body.get("stream"):
                    return self._stream(cid, model, messages, kwargs)
                resp = client.chat(messages, **kwargs)
            except RateLimitError as exc:
                return self._error(429, str(exc), "rate_limit")
            except FreeAIPIError as exc:
                return self._error(502, str(exc), "provider_error")
            except TypeError as exc:
                return self._error(400, str(exc), "invalid_request")
            msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
            if resp.tool_calls:
                msg["tool_calls"] = [{"id": t.id, "type": "function", "function": {
                    "name": t.name, "arguments": json.dumps(t.arguments)}} for t in resp.tool_calls]
            u = resp.usage
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": resp.model_id,
                "choices": [{"index": 0, "message": msg, "finish_reason": resp.finish_reason}],
                "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                          "total_tokens": u.total_tokens},
            })

        def _stream(self, cid: str, model: str, messages: Any, kwargs: dict[str, Any]) -> None:
            chunks = client.stream(messages, **kwargs)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(delta: dict[str, Any], finish: str | None = None) -> None:
                payload = {"id": cid, "object": "chat.completion.chunk",
                           "created": int(time.time()), "model": model or "auto",
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            try:
                for c in chunks:
                    if c.delta:
                        send({"content": c.delta})
                send({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
            except (FreeAIPIError, OSError):
                pass  # headers are already out; closing the stream is the signal

    return Handler


def make_server(client: FreeAIClient, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    """Build (but do not start) the gateway server."""
    return ThreadingHTTPServer((host, port), _make_handler(client))


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:  # pragma: no cover - blocking
    with FreeAIClient() as client:
        srv = make_server(client, host, port)
        print(f"freeaipi gateway on http://{host}:{port}/v1  ({len(client.registry)} models)  Ctrl+C to stop")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
