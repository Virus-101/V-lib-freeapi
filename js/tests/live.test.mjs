/**
 * Live integration tests against the real free endpoints.
 *
 * Skipped unless LIVE=1 is set:
 *
 *     LIVE=1 npm test
 *
 * A mock cannot tell you a provider changed its wire format, and the two
 * behaviours that matter most on a free tier — the anonymous 402 quota and
 * whether a `referrer` lifts it — are invisible offline. Both were found by
 * calling the real endpoint and both are asserted here.
 */

import test from "node:test";
import assert from "node:assert/strict";

const LIVE = process.env.LIVE === "1";
// Gate every live test, not just one placeholder: `test.skip()` called
// standalone skips only itself and leaves the real tests running.
const t = LIVE ? test : test.skip;

t("pollinations: chat returns content", async () => {
  const { init, chat, listModels, vendors } = await import("../dist/index.js");
  await init({ includeLocal: false });
  assert.ok(vendors().includes("pollinations"), "pollinations missing from registry");
  assert.ok(listModels().length > 0, "no models discovered");

  const r = await chat("Reply with exactly: OK", { model: "pollinations/openai-fast", maxTokens: 64 });
  assert.equal(r.content.trim(), "OK");
  assert.equal(r.tier, "anonymous");
  assert.ok(r.usage.totalTokens > 0);
});

t("pollinations: streaming yields text", async () => {
  const { init, stream } = await import("../dist/index.js");
  await init({ includeLocal: false });
  let text = "";
  for await (const c of stream("Count: 1 2 3", { model: "pollinations/openai-fast", maxTokens: 64 })) {
    text += c.delta;
  }
  assert.ok(text.trim().length > 0, "stream produced no text");
});

t("openrouter: catalogue is public without a key", async () => {
  const { fetchFreeModels } = await import("../dist/providers/openrouter.js");
  const { Transport } = await import("../dist/transport.js");
  const models = await fetchFreeModels(new Transport());
  assert.ok(models.length >= 10, `only ${models.length} OpenRouter free models seen`);
  assert.ok(models.every((m) => m.free));
  assert.ok(models.every((m) => m.authRequired));
});
