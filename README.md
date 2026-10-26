# freeaipi

One Python interface for **every free LLM API**. No paid tiers, no credit cards —
every model in the catalogue is either free, open-weight, or running on your own
hardware.

```python
from freeaipi import FreeAIClient

with FreeAIClient() as client:
    # No key, no configuration — works out of the box.
    reply = client.chat("Reply with exactly: OK")
    print(reply.content)          # OK
    print(reply.cost_usd)         # 0.0
    print(reply.usage.total_tokens)
```

### Or skip the client entirely

```python
import freeaipi

freeaipi.chat("Reply with exactly: OK").content      # one call, no setup
freeaipi.complete("Summarise: ...")
freeaipi.list_models()
freeaipi.best(requires_tools=True)
[c.delta for c in freeaipi.stream("Tell me a story")]
```

A single shared client is built on first use and reused, because construction
performs network discovery. Call `freeaipi.reset()` after changing credentials in
a long-lived process.

## Install

```bash
pip install freeaipi          # PyPI
pip install -e .              # this repo
```

Requires Python 3.10+. Only dependency is `httpx`.

### Node / TypeScript

The same library ships for JavaScript:

```bash
npm install freeaipi
```

```javascript
import { init, chat, listModels, vendors } from "freeaipi";

await init();                       // async: performs discovery
chat("Reply with exactly: OK", { model: "pollinations/openai-fast" }).then(r => {
  console.log(r.content);           // OK
  console.log(r.usage.totalTokens);
});
```

Requires Node 18+. Zero dependencies. The API mirrors the Python one — same
providers, same presets, same error classes:

```javascript
import { init, stream, runTools, listModels } from "freeaipi";
await init();

for await (const c of stream("Tell me a story", { maxTokens: 512 })) {
  if (c.delta) process.stdout.write(c.delta);
  else if (c.toolCall) console.error(c.toolCall.name);
}
```

Or the CLI:

```bash
npx freeaipi list --vendor groq
npx freeaipi chat "Reply with exactly: OK" --stream
```

Three kinds of backend, because "free" means three different things:

| Kind | Backend | Key | Limit |
|---|---|---|---|
| **Zero-key hosted** | Pollinations | none | 1 reply / 30 s per IP anonymous; a `referrer` lifts it |
| **Free-tier hosted** | OpenRouter `:free` | free account | per-key quota; 17+ models |
| **Local** | Ollama, llama.cpp, vLLM, LM Studio | none | none — your hardware |

### Pollinations (no key at all)

```
POST https://text.pollinations.ai/openai
```

OpenAI-shaped, answers with no credential, returns real token usage and a
`user_tier: "anonymous"` marker. The anonymous tier is quota-limited to one reply
per 30 seconds per IP, and it reports that as an HTTP 402 with the retry delay
in the body:

```json
{"status": 402, "error": {"code": "FREE_LIMIT",
 "message": "Free limit reached: 1 reply every 30 seconds per IP. Retry after 15s, ..."}}
```

`freeaipi` parses that into a `RateLimitError` carrying `retry_after`, and sends a
`referrer` by default, which was observed to lift the anonymous quota entirely.

### OpenRouter `:free`

`GET /api/v1/models` is public, so the catalogue is fetched without a key.
`POST /chat/completions` needs a free account key — free *money*, not free
*account*. Without a key the registry holds no OpenRouter entries and reports one
warning saying how many are known; a catalogue of models you cannot call is worse
than an honest message about what unlocks them.

Free models include `nvidia/nemotron-3-ultra-550b-a55b:free` (1M context),
`poolside/laguna-xs-2.1:free` and `cohere/north-mini-code:free`. The list changes
weekly, so it is fetched at runtime rather than hardcoded.

### Any OpenAI-compatible endpoint (config, not code)

Nearly every vendor speaks the same protocol — `POST {base_url}/chat/completions`
with a bearer token, plus `GET {base_url}/models` for discovery. A class per
vendor would mean fifteen copies of identical code, so there is **one generic
provider driven by a spec**. Adding a free tier is a config entry, not a library
change.

Curated presets ship for the known free tiers:

| Preset | Env var | Free tier | Notes |
|---|---|---|---|
| `groq` | `GROQ_API_KEY` | yes | fastest inference; Llama, Qwen, Gemma, Mistral |
| `cerebras` | `CEREBRAS_API_KEY` | yes | very high throughput |
| `sambanova` | `SAMBANOVA_API_KEY` | yes | DeepSeek-V3, Llama-3.3, MiniMax; catalogue is public |
| `together` | `TOGETHER_API_KEY` | grant | free credit on signup, then paid |
| `fireworks` | `FIREWORKS_API_KEY` | yes | per-day token grant |
| `mistral` | `MISTRAL_API_KEY` | yes | free plan excludes flagship models |
| `novita` | `NOVITA_API_KEY` | yes | DeepSeek, Llama, Qwen |
| `deepseek` | `DEEPSEEK_API_KEY` | **no** | priced; marked `free=False` deliberately |
| `nous` | `NOUS_API_KEY` | **no** | priced; what Hermes uses |
| `inception` | `INCEPTION_API_KEY` | **no** | priced |

**Auto-enable:** a preset appears when its env var is set. No config file, no code
change — set a key and the provider shows up in `list_models()`.

```bash
export GROQ_API_KEY=gsk_...
freeaipi list --vendor groq
```

**Custom endpoints** via `FREEAIPI_PROVIDERS` pointing at a JSON file:

```json
{"providers": [
  {"namespace": "myvllm", "base_url": "http://127.0.0.1:8000/v1",
   "key_env": "MYVLLM_KEY", "free": true},
  {"namespace": "private", "base_url": "http://10.0.0.5:8080/v1",
   "free": true, "models": ["my-secret-70b"]}
]}
```

The second entry has no `key_env`, so no auth header is sent — a private server
with no `/models` endpoint works by listing ids explicitly.

**Honesty about `free`:** the flag means the vendor publishes a no-cost tier, not
that it is unlimited; the `rate_limit` field says the real constraint. Three
presets are included *specifically to be marked not free* (`deepseek`, `nous`,
`inception`) so the catalogue is never misread as a promise. `verified` records
whether the endpoint was actually called during development rather than assumed.

### Local

Ollama, llama.cpp, vLLM, LM Studio, Jan and GPT4All all speak the OpenAI chat
protocol on localhost, so they share one implementation. Ollama's native
`/api/chat` NDJSON is also supported, because it exposes `num_ctx` and other
options the OpenAI-compat endpoint does not.

Discovery probes the well-known ports at client construction. A dead port is
skipped silently — on a normal desktop four of five are always down.

## Install

```bash
pip install freeaipi          # from PyPI, once published
pip install -e .              # from this repo
```

Requires Python 3.10+. Only dependency is `httpx`.

## Usage

### Choose a model

```python
client.chat("hi")                                  # auto-pick
client.chat("hi", model="pollinations/openai-fast")
client.chat("hi", tools=[...])                     # auto-pick one that can call tools
```

`best()` is deterministic and prefers a model that needs no key, then the largest
context window:

```python
client.best()                              # 'local/llama3' (no key, biggest ctx)
client.best(requires_tools=True)
client.best(prefer_no_key=False)           # allow keyed models
```

Registry keys are `provider/model`. A bare model id works when unique; an
ambiguous one raises and lists the options rather than guessing.

### Stream

```python
for chunk in client.stream("Tell me a story", max_tokens=512):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    elif chunk.reasoning:
        pass          # chain-of-thought arrives separately, not glued to the answer
    elif chunk.tool_call:
        print(f"calling {chunk.tool_call['name']}")
```

### Tool calling

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

# Blocking
reply = client.chat("Weather in Paris?", tools=tools)
reply.tool_calls[0].name        # 'get_weather'
reply.tool_calls[0].arguments   # {'city': 'Paris'}

# Or drive the loop yourself
result = client.run_tools(
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    handlers={"get_weather": lambda city: f"{city}: 22C, clear"},
)
```

`run_tools` executes every requested tool, appends results as `tool` messages, and
repeats until the model stops calling tools. A handler that raises becomes an
`error: ...` tool result rather than killing the turn. Hitting `max_rounds` is
recorded in `result.raw["round_cap_reached"]` — a silent stop at the cap looks
exactly like a natural finish otherwise.

### Fallback

```python
client.chat("hi", model="pollinations/openai-fast",
            fallback=["openrouter/nvidia/nemotron-3:free"])
```

A quota error is **not** fallen back from: switching models to dodge a rate limit
hides the problem from the caller, who then cannot back off correctly.

### Credentials

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export POLLINATIONS_API_KEY=...     # optional; lifts the anonymous quota
```

```python
FreeAIClient(openrouter_key="sk-or-v1-...")   # explicit beats env
```

Keys are resolved per request, never captured at construction — a token read once
at startup is a bug that only appears an hour later. Logs and `status()` report
credential *presence*, never the value:

```python
client.status()["credentials"]   # {'openrouter': '<absent>', 'pollinations': '<set len=40 prefix=sk-o…>'}
```

### Catalogue

```python
client.list_models()                                # everything registered
client.list_models(provider=Provider.OPENROUTER)
client.list_models(namespace="groq")                # one vendor
client.for_vendor("groq")
client.vendors()                                    # ['groq', 'pollinations', 'local']
client.list_models(requires_tools=True)
client.list_models(requires_vision=True)
client.no_key_models()                              # usable with no credential
client.describe("pollinations/openai-fast")
client.warnings                                     # non-fatal discovery problems
```

## CLI

```bash
freeaipi list                    # every model usable with no key
freeaipi list --no-key-only      # include keyed models
freeaipi list --provider openrouter --json

freeaipi chat "Reply with exactly: OK"
freeaipi chat "Weather in Paris?" --tools --max-tokens 512
freeaipi chat "hi" --stream
freeaipi chat "hi" --json        # full response incl. usage and reasoning

freeaipi check                   # registry summary + credential presence
```

## Errors

Every error is a subclass of `FreeAIPIError`. The subclasses exist because the
handling differs in kind:

| Error | Meaning | Fix |
|---|---|---|
| `ConfigurationError` | credential/setting missing | set the env var, start Ollama |
| `AuthenticationError` | 401/403 — key rejected | fix the key |
| `RateLimitError` | 402/429 — quota | sleep `retry_after`, then retry |
| `ProviderError` | other non-success | read `status_code` / `body` |
| `ModelNotFoundError` | id not in registry | check the message for valid keys |

## Testing

```bash
# Python — 232 tests, no network
py -3.12 -m pytest tests/
LIVE=1 py -3.12 -m pytest tests/test_live.py     # real endpoints

# Node — 53 tests, no network
cd js && npm test
cd js && LIVE=1 npm test                         # real endpoints
```

The offline suites inject a fake transport instead of monkeypatching `httpx` /
`fetch`, so provider request-shaping runs exactly as it ships. The live suites
exist because a mock cannot tell you a provider changed its wire format — and
the two behaviours that matter most here, the anonymous 402 quota and whether a
`referrer` lifts it, are invisible offline. Both were found by calling the real
endpoint and both are asserted in `tests/test_live.py` and
`js/tests/live.test.mjs`.

The JS live suite is what caught a real streaming bug: `streamLines` yielded raw
decoded chunks, so an SSE frame split across two reads never parsed and the
caller received no text at all — with no error. It now buffers and emits
complete lines.

## Scope, honestly

- **Not covered:** paid APIs. Nothing here charges money, which is the point.
- **Local models are not downloaded.** `freeaipi` calls whatever your local server
  is serving; it does not fetch weights. Ollama must be running for local models
  to appear.
- **Capability metadata for local models is inferred from the name.** Reading it
  properly means parsing the GGUF header. The values are a hint, marked as such.
- **The free catalogues change.** Pollinations and OpenRouter are fetched at
  runtime so new models appear without a library change; a network outage during
  discovery leaves a reduced catalogue plus a warning, never a crash.

## Licence

MIT
