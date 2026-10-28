"""Command-line interface.

Lists the catalogue and runs one-shot completions. Useful on its own, and it is
how the live endpoints are verified during development without writing a script
each time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .client import FreeAIClient
from .errors import FreeAIPIError


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fllm",
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

    sub.add_parser("init", help="Set up your own free keys, step by step.")

    key = sub.add_parser("key", help="Manage the keys stored on this machine.")
    ksub = key.add_subparsers(dest="key_cmd", required=True)
    kset = ksub.add_parser("set", help="Store a key (prompts, so it stays out of shell history).")
    kset.add_argument("vendor", help="e.g. groq, openrouter, cerebras")
    ksub.add_parser("list", help="Show which keys are stored (never the values).")
    krm = ksub.add_parser("remove", help="Delete a stored key.")
    krm.add_argument("vendor")

    use = sub.add_parser("usage", help="Tokens and calls you have used, from the local ledger.")
    use.add_argument("--days", type=float, default=0, help="Only the last N days.")
    use.add_argument("--json", action="store_true")

    srv = sub.add_parser("serve", help="Run a local OpenAI-compatible gateway.")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8787)
    return p


def _env_for(vendor: str) -> str | None:
    from . import home

    names = home.key_names()
    if vendor in names:
        return names[vendor]
    return vendor if vendor.isupper() and vendor.endswith("_KEY") else None


def _local_command(args: argparse.Namespace) -> int:
    import getpass
    import time

    from . import home

    if args.cmd == "usage":
        since = time.time() - args.days * 86400 if args.days else 0.0
        summary = home.usage_summary(since)
        if args.json:
            print(json.dumps(summary, indent=2))
            return 0
        print(f"{summary['calls']} call(s), {summary['tokens']:,} token(s)")
        for model, row in sorted(summary["models"].items(), key=lambda kv: -kv[1]["tokens"]):
            print(f"  {model:52s} {row['calls']:6d} calls {row['tokens']:12,d} tok")
        return 0

    if args.cmd == "serve":
        from .gateway import serve

        serve(args.host, args.port)
        return 0

    if args.cmd == "key":
        if args.key_cmd == "list":
            stored = home.read_keys()
            for vendor, var in home.key_names().items():
                state = "stored" if var in stored else ("env" if os.environ.get(var) else "-")
                print(f"{vendor:12s} {var:22s} {state}")
            return 0
        var = _env_for(args.vendor)
        if not var:
            print(f"unknown vendor {args.vendor!r}; try: {', '.join(home.key_names())}", file=sys.stderr)
            return 2
        if args.key_cmd == "remove":
            print("removed" if home.remove_key(var) else "no such stored key")
            return 0
        value = getpass.getpass(f"{var} (input hidden): ").strip()
        if not value:
            print("nothing entered", file=sys.stderr)
            return 2
        print(f"saved to {home.set_key(var, value)}")
        return 0

    # init: no key is needed to start; free accounts add models and quota.
    print("freeaipi setup. Keys stay on this machine in", home.keys_path())
    print("Pollinations and local models work with no key. Press Enter to skip any vendor.\n")
    stored = home.read_keys()
    for vendor, var in home.key_names().items():
        if var in stored or os.environ.get(var):
            print(f"{vendor}: already set")
            continue
        print(f"{vendor}: get a free key at {home.SIGNUP.get(vendor, '(vendor site)')}")
        value = getpass.getpass(f"  {var}: ").strip()
        if value:
            home.set_key(var, value)
    print('\nDone. Try:  fllm chat "hello"   |   fllm list   |   fllm serve')
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd in ("init", "key", "usage", "serve"):
        return _local_command(args)
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
