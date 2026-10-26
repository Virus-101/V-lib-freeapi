/**
 * freeaipi JS tests. Run with `npm test` (node --test).
 *
 * Transport tests inject a fake `fetch` rather than monkeypatching globals, so
 * request-shaping runs exactly as it ships.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { Transport } from "../dist/transport.js";
import { Provider, toolCallFromOpenAI, usageFromOpenAI } from "../dist/models.js";
import {
  AuthenticationError,
  ConfigurationError,
  ModelNotFoundError,
  ProviderError,
  RateLimitError,
} from "../dist/errors.js";
import { PRESETS, enabledSpecs, loadConfigFile, configPaths } from "../dist/presets.js";
import {
  OpenAICompatProvider,
  fetchCatalogue,
  parseCatalogue,
  safeJson,
} from "../dist/providers/openai_compat.js";
import {
  PollinationsProvider,
  flattenContent,
  pollinationsUrl,
  quotaError,
} from "../dist/providers/pollinations.js";
import { createDefaultRegistry } from "../dist/registry.js";

// Credentials for the key_envs the registry tests declare. KEYLESS is
// deliberately absent: one test asserts the missing-key path.
for (const n of ["TESTVENDOR", "ALPHA", "BETA", "V", "GROQ", "PRIVATE"]) {
  if (!process.env[`${n}_API_KEY`]) process.env[`${n}_API_KEY`] = `${n.toLowerCase()}-key`;
}

// ── helpers ────────────────────────────────────────────────────────────────

function spec(namespace = "testvendor", extra = {}) {
  return {
    namespace,
    baseUrl: "https://api.example.com/v1",
    keyEnv: extra.keyEnv ?? `${namespace.toUpperCase()}_API_KEY`,
    free: true,
    rateLimit: "",
    notes: "",
    verified: false,
    ...extra,
  };
}

function jsonResponse(status, body, headers = {}) {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

/** A Transport with a scripted fetch. The handler sees url, headers and parsed body. */
function scripted(handler, opts = {}) {
  return new Transport({
    maxRetries: 0,
    fetchFn: (async (input, init) => {
      const url = String(input);
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      const headers = init?.headers ?? {};
      return jsonResponse(200, await handler(url, { headers, body }));
    }),
    ...opts,
  });
}

function chatResponse(content = "OK", extra = {}) {
  return {
    choices: [{ message: { content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    ...extra,
  };
}

function compatModel(over = {}) {
  return {
    modelId: "test-model-70b",
    provider: Provider.Compat,
    providerModelId: "test-model-70b",
    name: "test-model-70b",
    capabilities: { text: true, vision: false, tools: false, reasoning: false, jsonMode: false, contextLength: 65536, maxOutputTokens: 4096 },
    free: true,
    authRequired: true,
    rateLimit: "",
    notes: "",
    ...over,
  };
}

// ── transport ──────────────────────────────────────────────────────────────

test("transport: success returns decoded body", async () => {
  const t = scripted(() => ({ ok: true }));
  assert.deepEqual(await t.postJson("http://x", {}), { ok: true });
});

test("transport: retries transient then succeeds", async () => {
  let calls = 0;
  const t = new Transport({
    maxRetries: 3,
    retryBackoffMs: 1,
    fetchFn: async () => {
      calls++;
      return calls < 3 ? jsonResponse(503, "overloaded") : jsonResponse(200, { ok: true });
    },
  });
  assert.deepEqual(await t.postJson("http://x", {}), { ok: true });
  assert.equal(calls, 3);
});

test("transport: a 500 exhausting retries is ProviderError, not RateLimitError", async () => {
  // Routing a persistent 500 into the rate-limit branch sends the caller off to
  // sleep on a failure that will never succeed.
  let calls = 0;
  const t = new Transport({
    maxRetries: 2,
    retryBackoffMs: 1,
    fetchFn: async () => {
      calls++;
      return jsonResponse(500, "boom");
    },
  });
  await assert.rejects(() => t.postJson("http://x", {}), ProviderError);
  assert.equal(calls, 3);
});

test("transport: 429 raises RateLimitError with retryAfter from the header", async () => {
  const t = new Transport({
    maxRetries: 1,
    retryBackoffMs: 1,
    fetchFn: async () => jsonResponse(429, "slow down", { "retry-after": "7" }),
  });
  await t.postJson("http://x", {}).then(
    () => assert.fail("should have thrown"),
    (e) => {
      assert.ok(e instanceof RateLimitError);
      assert.equal(e.retryAfter, 7);
    },
  );
});

test("transport: retry hint parsed from body text, not only the header", async () => {
  let calls = 0;
  const t = new Transport({
    maxRetries: 2,
    retryBackoffMs: 1,
    fetchFn: async () => {
      calls++;
      return calls === 1
        ? jsonResponse(402, JSON.stringify({ status: 402, error: { message: "Free limit reached. Retry after 15s" } }))
        : jsonResponse(200, { ok: 1 });
    },
  });
  assert.deepEqual(await t.postJson("http://x", {}), { ok: 1 });
  assert.equal(calls, 2);
});

test("transport: 401 raises AuthenticationError and is not retried", async () => {
  let calls = 0;
  const t = new Transport({
    maxRetries: 3,
    retryBackoffMs: 1,
    fetchFn: async () => {
      calls++;
      return jsonResponse(401, "bad key");
    },
  });
  await assert.rejects(() => t.postJson("http://x", {}), AuthenticationError);
  assert.equal(calls, 1);
});

test("transport: non-JSON 200 raises ProviderError", async () => {
  const t = new Transport({
    maxRetries: 0,
    fetchFn: async () => new Response("<html>nope</html>", { status: 200 }),
  });
  await assert.rejects(() => t.getJson("http://x"), ProviderError);
});

test("transport: 404 carries status and body", async () => {
  const t = new Transport({ maxRetries: 0, fetchFn: async () => jsonResponse(404, "not found here") });
  await t.getJson("http://x").then(
    () => assert.fail("should have thrown"),
    (e) => {
      assert.ok(e instanceof ProviderError);
      assert.equal(e.statusCode, 404);
      assert.match(e.body ?? "", /not found here/);
    },
  );
});

// ── catalogue parsing ──────────────────────────────────────────────────────

test("parseCatalogue: openai shape", () => {
  const models = parseCatalogue({ data: [{ id: "m1", context_length: 32768, max_completion_tokens: 4096 }] }, spec());
  assert.equal(models.length, 1);
  assert.equal(models[0].modelId, "m1");
  assert.equal(models[0].capabilities.contextLength, 32768);
});

test("parseCatalogue: bare list shape", () => {
  assert.equal(parseCatalogue([{ id: "m1" }], spec()).length, 1);
});

test("parseCatalogue: skips entries without id", () => {
  assert.deepEqual(parseCatalogue([{ id: "good" }, { context_length: 8 }, "nope", { id: "also" }], spec()).map((m) => m.modelId), ["good", "also"]);
});

test("parseCatalogue: odd shapes do not throw", () => {
  for (const raw of [null, {}, { data: null }, { data: "x" }, 42, "s"]) {
    assert.deepEqual(parseCatalogue(raw, spec()), []);
  }
});

test("parseCatalogue: namespace sets the key while kind stays Compat", () => {
  const m = parseCatalogue([{ id: "m1" }], spec("groq"))[0];
  assert.equal(m.namespace, "groq");
  assert.equal(m.provider, Provider.Compat);
  assert.equal(`${m.namespace}/${m.modelId}`, "groq/m1");
});

test("parseCatalogue: capabilities from modalities and supported_parameters", () => {
  const m = parseCatalogue(
    [{ id: "m1", architecture: { input_modalities: ["text", "image"] }, supported_parameters: ["tools", "reasoning"] }],
    spec(),
  )[0];
  assert.equal(m.capabilities.vision, true);
  assert.equal(m.capabilities.tools, true);
  assert.equal(m.capabilities.reasoning, true);
});

test("parseCatalogue: capabilities fall back to the name", () => {
  const m = parseCatalogue([{ id: "qwen2.5-vl-coder-72b" }], spec())[0];
  assert.equal(m.capabilities.vision, true);
  assert.equal(m.capabilities.tools, true);
});

test("parseCatalogue: free flag carried from spec, authRequired always true", () => {
  assert.equal(parseCatalogue([{ id: "m" }], spec("v", { free: false }))[0].free, false);
  assert.equal(parseCatalogue([{ id: "m" }], spec())[0].authRequired, true);
});

// ── provider requests ──────────────────────────────────────────────────────

test("compat provider: url, auth header and wire model id", async () => {
  let seen = null;
  const t = scripted((url, init) => {
    seen = { url, headers: init.headers, body: init.body };
    return chatResponse();
  });
  const p = new OpenAICompatProvider(compatModel(), t, spec(), "k");
  await p.chat({ messages: [{ role: "user", content: "hi" }] });
  assert.equal(seen.url, "https://api.example.com/v1/chat/completions");
  assert.equal(seen.headers.authorization, "Bearer k");
  assert.equal(seen.body.model, "test-model-70b");
});

test("compat provider: bare string message coerced, not iterated", async () => {
  let body = null;
  const t = scripted((_url, init) => {
    body = init.body;
    return chatResponse();
  });
  const p = new OpenAICompatProvider(compatModel(), t, spec(), "k");
  // Providers are reachable via registry.get(), so the coercion belongs here.
  await p.chat({ messages: "hi" });
  assert.deepEqual(body.messages, [{ role: "user", content: "hi" }]);
});

test("compat provider: no auth header when the spec declares none", async () => {
  let headers = {};
  const t = scripted((_url, init) => {
    headers = init.headers;
    return chatResponse();
  });
  const s = spec("private", { keyEnv: "" });
  const p = new OpenAICompatProvider(compatModel({ namespace: "private" }), t, s, "");
  await p.chat({ messages: [{ role: "user", content: "hi" }] });
  assert.equal(headers.authorization, undefined);
});

test("compat provider: missing key rejected only when the spec declares one", () => {
  assert.throws(() => new OpenAICompatProvider(compatModel(), scripted(() => ({}), {}), spec(), ""), ConfigurationError);
});

test("compat provider: tools, json mode and seed forwarded", async () => {
  let body = {};
  const t = scripted((_url, init) => {
    body = init.body;
    return chatResponse("");
  });
  const tools = [{ type: "function", function: { name: "f", parameters: { type: "object" } } }];
  const p = new OpenAICompatProvider(compatModel(), t, spec(), "k");
  await p.chat({ messages: [{ role: "user", content: "hi" }], tools, jsonMode: true, seed: 42 });
  assert.deepEqual(body.tools, tools);
  assert.equal(body.tool_choice, "auto");
  assert.deepEqual(body.response_format, { type: "json_object" });
  assert.equal(body.seed, 42);
});

test("compat provider: parses content, usage, reasoning and tool calls", async () => {
  const t = scripted(() => ({
    choices: [{
      message: {
        content: "answer",
        reasoning: "think",
        tool_calls: [{ id: "c1", function: { name: "f", arguments: '{"a":1}' } }],
      },
      finish_reason: "tool_calls",
    }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }));
  const p = new OpenAICompatProvider(compatModel(), t, spec(), "k");
  const r = await p.chat({ messages: [{ role: "user", content: "hi" }] });
  assert.equal(r.content, "answer");
  assert.equal(r.reasoning, "think");
  assert.equal(r.usage.totalTokens, 5);
  assert.equal(r.toolCalls[0].arguments.a, 1);
  assert.equal(r.finishReason, "tool_calls");
});

test("toolCallFromOpenAI: malformed arguments surface as _raw", () => {
  const tc = toolCallFromOpenAI({ id: "c", function: { name: "f", arguments: "{broken" } });
  assert.deepEqual(tc.arguments, { _raw: "{broken" });
});

test("compat provider: error envelope raises ProviderError", async () => {
  const t = scripted(() => ({ error: { message: "bad", code: 400 } }));
  const p = new OpenAICompatProvider(compatModel(), t, spec(), "k");
  await assert.rejects(() => p.chat({ messages: [{ role: "user", content: "hi" }] }), ProviderError);
});

// ── pollinations ───────────────────────────────────────────────────────────

test("pollinations: url is /openai, not the root", () => {
  assert.equal(pollinationsUrl(), "https://text.pollinations.ai/openai");
  assert.equal(pollinationsUrl("https://x/"), "https://x/openai");
});

test("pollinations: quota 402 parsed with retry hint", () => {
  const err = quotaError(JSON.stringify({ status: 402, error: { message: "Free limit reached. Retry after 15s" } }));
  assert.ok(err instanceof RateLimitError);
  assert.equal(err.retryAfter, 15);
});

test("pollinations: content part-list flattened", () => {
  assert.equal(flattenContent([{ type: "text", text: "ab" }, { type: "text", text: "cd" }]), "abcd");
  assert.equal(flattenContent("plain"), "plain");
});

test("pollinations: referrer sent by default, no auth header without a key", async () => {
  let body = {};
  let headers = {};
  const t = scripted((_url, init) => {
    body = init.body;
    headers = init.headers;
    return chatResponse("OK", { user_tier: "anonymous" });
  });
  const model = compatModel({ modelId: "openai-fast", provider: Provider.Pollinations, providerModelId: "gpt-oss-20b", authRequired: false });
  const p = new PollinationsProvider(model, t, "");
  const r = await p.chat({ messages: [{ role: "user", content: "hi" }] });
  assert.equal(body.referrer, "freeaipi");
  assert.equal(headers.authorization, undefined);
  assert.equal(r.tier, "anonymous");
  assert.equal(r.content, "OK");
});

test("pollinations: sends auth header when a key is present", async () => {
  let headers = {};
  const t = scripted((_url, init) => {
    headers = init.headers;
    return chatResponse();
  });
  const model = compatModel({ modelId: "openai-fast", provider: Provider.Pollinations, providerModelId: "gpt-oss-20b", authRequired: false });
  const p = new PollinationsProvider(model, t, "pollen-abc");
  await p.chat({ messages: [{ role: "user", content: "hi" }] });
  assert.equal(headers.authorization, "Bearer pollen-abc");
});

test("safeJson: valid, empty, malformed, non-object", () => {
  assert.deepEqual(safeJson('{"a":1}'), { a: 1 });
  assert.deepEqual(safeJson(""), {});
  assert.deepEqual(safeJson("{broken"), { _raw: "{broken" });
  assert.deepEqual(safeJson("[1]"), { _raw: "[1]" });
});

test("usageFromOpenAI: maps the OpenAI triple", () => {
  assert.deepEqual(usageFromOpenAI({ prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 }), { promptTokens: 1, completionTokens: 2, totalTokens: 3 });
  assert.deepEqual(usageFromOpenAI(undefined), { promptTokens: 0, completionTokens: 0, totalTokens: 0 });
});

// ── presets ────────────────────────────────────────────────────────────────

test("presets: env var enables a provider", () => {
  assert.deepEqual(enabledSpecs({ GROQ_API_KEY: "x" }).map((s) => s.namespace), ["groq"]);
});

test("presets: no env vars enables none", () => {
  assert.deepEqual(enabledSpecs({}), []);
});

test("presets: whitespace is not a key", () => {
  assert.deepEqual(enabledSpecs({ GROQ_API_KEY: "   " }), []);
});

test("presets: every preset has a distinct namespace and base url", () => {
  const all = Object.values(PRESETS);
  assert.equal(new Set(all.map((s) => s.namespace)).size, all.length);
  assert.ok(all.every((s) => s.baseUrl.startsWith("http")));
});

test("presets: paid providers are marked not free", () => {
  for (const name of ["deepseek", "nous", "inception"]) {
    assert.equal(PRESETS[name].free, false, `${name} should be free=false`);
  }
});

test("presets: free providers are marked free", () => {
  for (const name of ["groq", "cerebras", "sambanova", "novita"]) {
    assert.equal(PRESETS[name].free, true, `${name} should be free=true`);
  }
});

test("presets: config file loads, skips bad entries, tolerates malformed JSON", () => {
  const specs = loadConfigFile("p.json", () => JSON.stringify({ providers: [{ namespace: "v", baseUrl: "http://x:1/v1", free: true }] }));
  assert.equal(specs[0].namespace, "v");

  const mixed = loadConfigFile("p.json", () => JSON.stringify({ providers: [{ namespace: "no-url" }, { namespace: "w", baseUrl: "http://w" }] }));
  assert.deepEqual(mixed.map((s) => s.namespace), ["w"]);

  assert.deepEqual(loadConfigFile("bad.json", () => "{not json"), []);
});

test("presets: configPaths ignores empty values", () => {
  assert.deepEqual(configPaths({}), []);
  assert.deepEqual(configPaths({ FREEAIPI_PROVIDERS: "  " }), []);
  assert.deepEqual(configPaths({ FREEAIPI_PROVIDERS: "/tmp/a.json" }), ["/tmp/a.json"]);
});

// ── registry ───────────────────────────────────────────────────────────────

test("registry: spec models used when the catalogue is unreachable", async () => {
  const t = scripted(() => {
    throw new Error("no catalogue endpoint");
  });
  const s = spec("private", { keyEnv: "", models: ["my-70b"] });
  const reg = await createDefaultRegistry(t, {
    includeLocal: false, includeOpenrouter: false, includePollinations: false, compatSpecs: [s],
  });
  assert.deepEqual([...reg.models.keys()], ["private/my-70b"]);
});

test("registry: models filter narrows, exclude removes", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [{ id: "m1" }, { id: "m2" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false,
    compatSpecs: [spec("v", { models: ["m2"] })],
  });
  assert.deepEqual([...reg.models.keys()], ["v/m2"]);

  const reg2 = await createDefaultRegistry(scripted(() => ({ data: [{ id: "m1" }, { id: "m2" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false,
    compatSpecs: [spec("v", { exclude: ["m1"] })],
  });
  assert.deepEqual([...reg2.models.keys()], ["v/m2"]);
});

test("registry: a missing key warns and other providers survive", async () => {
  const t = scripted(() => ({ data: [{ id: "good" }] }));
  const reg = await createDefaultRegistry(t, {
    includeLocal: false, includeOpenrouter: false, includePollinations: false,
    // keyless has no key (KEYLESS_KEY unset); alpha resolves via the fixture.
    compatSpecs: [spec("keyless", { keyEnv: "KEYLESS_KEY" }), spec("alpha")],
  });
  assert.deepEqual([...reg.models.keys()], ["alpha/good"]);
  assert.ok(reg.warnings.some((w) => w.includes("keyless")));
});

test("registry: empty catalogue warns", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false, compatSpecs: [spec("v")],
  });
  assert.ok(reg.warnings.some((w) => w.includes("no models discovered")));
});

test("registry: namespace filter selects a vendor", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [{ id: "a" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false, compatSpecs: [spec("alpha")],
  });
  assert.deepEqual(reg.list({ namespace: "alpha" }).map((m) => m.modelId), ["a"]);
  assert.deepEqual(reg.list({ namespace: "gamma" }), []);
});

test("registry: get resolves vendor key and bare unique id; unknown raises", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [{ id: "m1" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false, compatSpecs: [spec("groq")],
  });
  assert.equal(reg.get("groq/m1").model.modelId, "m1");
  assert.equal(reg.get("m1").model.modelId, "m1");
  assert.throws(() => reg.get("nope"), ModelNotFoundError);
});

test("registry: providersInUse lists namespaces", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [{ id: "a" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false,
    compatSpecs: [spec("alpha"), spec("beta")],
  });
  assert.deepEqual(reg.providersInUse(), ["alpha", "beta"]);
});

test("registry: includeCompat false skips compat providers", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [{ id: "a" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false,
    includeCompat: false, compatSpecs: [spec("v")],
  });
  assert.equal(reg.size, 0);
});

test("registry: unknown provider kind raises with the known list", async () => {
  const reg = await createDefaultRegistry(scripted(() => ({ data: [{ id: "a" }] })), {
    includeLocal: false, includeOpenrouter: false, includePollinations: false,
    includeCompat: false, compatSpecs: [],
  });
  assert.throws(() => reg.list({ provider: "nope" }), ModelNotFoundError);
});

// ── fetchCatalogue ─────────────────────────────────────────────────────────

test("fetchCatalogue: uses baseUrl/models and sends auth when given", async () => {
  let seen = null;
  const t = scripted((url, init) => {
    seen = { url, headers: init.headers };
    return { data: [{ id: "m1", context_length: 8000 }] };
  });
  const models = await fetchCatalogue(t, spec(), "k");
  assert.equal(seen.url, "https://api.example.com/v1/models");
  assert.equal(seen.headers.authorization, "Bearer k");
  assert.deepEqual(models.map((m) => m.modelId), ["m1"]);
});

test("fetchCatalogue: failure returns empty, does not throw", async () => {
  assert.deepEqual(await fetchCatalogue(scripted(() => { throw new Error("offline"); }), spec(), "k"), []);
});
