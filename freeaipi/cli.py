"""Command-line interface.

Lists the catalogue and runs one-shot completions. Useful on its own, and it is
how the live endpoints are verified during development without writing a script
each time.
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import FreeAIClient
from .errors import FreeAIPIError


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="freeaipi",
        description="Call every free LLM API from one interface.",
    )
    p.add_argument("--key", dest="key_name", choices=["openrouter", "pollinations"],
                   help="Read the key from an env var (default: both, if set).")
    p.add_argument("--no-local", action="store_true", help="Skip local backend discovery.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="List every known free model.")
    ls.add_argument("--provider", default=None, help="Filter by provider kind.")
    ls.add_argument("--vendor", default=None, help="Filter by vendor (groq, cerebras…).")
    ls.add_argument("--json", action="store_true", help="Emit JSON.")
    ls.add_argument("--no-key-only", action="store_true",
                    help="Include models that require an account key.")

    chat = sub.add_parser("chat", help="Send one message and print the reply.")
    chat.add_argument("message", nargs="?", default=None)
    chat.add_argument("--model", "-m", default=None, help="Model key (default: auto-pick).")
    chat.add_argument("--system", "-s", default=None)
    chat.add_argument("--max-tokens", type=int, default=2048)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--seed", type=int, default=None)
    chat.add_argument("--fallback", action="append", default=[],
                       help="Model key to try if the first fails; repeatable.")
    chat.add_argument("--tools", action="store_true", help="Declare a demo tool.")
    chat.add_argument("--json", action="store_true", help="Emit the full response as JSON.")
    chat.add_argument("--stream", action="store_true", help="Stream instead of blocking.")

    chk = sub.add_parser("check", help="Report credential presence and reachability.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = None
    try:
        # Construction is inside the try: a missing credential raises
        # ConfigurationError, and a traceback is the wrong answer for that.
        client = FreeAIClient(include_local=not getattr(args, "no_local", False))
        if args.cmd == "list":
            filters = {}
            if args.provider:
                filters["provider"] = args.provider
            if args.vendor:
                filters["namespace"] = args.vendor
            models = client.list_models(**filters)
            if not args.no_key_only:
                models = [m for m in models if not m.auth_required]
            if args.json:
                print(json.dumps([m.to_dict() for m in models], indent=2))
            else:
                for m in sorted(models, key=lambda x: (x.label, x.model_id)):
                    caps = m.capabilities
                    flags = "".join(
                        c for c, on in (("T", caps.tools), ("V", caps.vision), ("R", caps.reasoning)) if on
                    ) or "-"
                    key = "key" if m.auth_required else "no-key"
                    print(f"{m.key:52s} [{flags}] {key:6s} {m.name}")
                scope = f" (filtered)" if (args.provider or args.vendor) else ""
                print(f"\n{len(models)} model(s){scope} across "
                      f"{len(set(m.label for m in models)) or 0} of "
                      f"{len(client.vendors())} vendor(s). "
                      f"Warnings: {client.warnings or 'none'}")
            return 0

        if args.cmd == "chat":
            message = args.message or sys.stdin.read().strip()
            if not message:
                print("No message given.", file=sys.stderr)
                return 2
            kwargs = {
                "model": args.model,
                "system": args.system,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "fallback": args.fallback,
            }
            if args.stream:
                for chunk in client.stream(message, **kwargs):
                    if chunk.delta:
                        print(chunk.delta, end="", flush=True)
                    elif chunk.tool_call:
                        print(f"\n[tool call] {json.dumps(chunk.tool_call)}")
                print()
                return 0
            tools = None
            if args.tools:
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Return the current time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }]
            resp = client.chat(message, tools=tools, **kwargs)
            if args.json:
                print(json.dumps(resp.to_dict(), indent=2))
            else:
                print(resp.content)
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        print(f"[tool call] {tc.name}({json.dumps(tc.arguments)})", file=sys.stderr)
                print(
                    f"\n— {resp.model_id} · {resp.latency_ms:.0f} ms · "
                    f"{resp.usage.total_tokens} tok · ${resp.cost_usd:.4f}",
                    file=sys.stderr,
                )
            return 0

        if args.cmd == "check":
            print(json.dumps(client.status(), indent=2))
            return 0
    except FreeAIPIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Construction can fail before the client exists.
        if client is not None:
            client.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
