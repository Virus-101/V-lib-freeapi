#!/usr/bin/env node
/**
 * CLI: lists the catalogue and runs one-shot completions.
 *
 *   freeaipi list [--vendor groq] [--no-key-only] [--json]
 *   freeaipi chat "Reply with exactly: OK" [--model pollinations/openai-fast]
 *   freeaipi check
 */

import { FreeAIClient, Provider } from "./index.js";
import type { ModelInfo } from "./index.js";

function flags(argv: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (!a.startsWith("--")) continue;
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      out[a] = next;
      i++;
    } else {
      out[a] = true;
    }
  }
  return out;
}

function modelKey(m: ModelInfo): string {
  return `${m.namespace ?? m.provider}/${m.modelId}`;
}

const [cmd, ...rest] = process.argv.slice(2);
const opts = flags(rest);

const client = await FreeAIClient.create({ includeLocal: !opts["--no-local"] });

if (cmd === "list") {
  let models = client.listModels();
  if (opts["--vendor"]) models = client.forVendor(String(opts["--vendor"]));
  if (opts["--provider"]) {
    const want = String(opts["--provider"]);
    models = client.listModels({ provider: want as Provider });
  }
  if (!opts["--no-key-only"]) models = client.noKeyModels();

  if (opts["--json"]) {
    console.log(JSON.stringify(models.map((m) => ({ ...m, key: modelKey(m) })), null, 2));
  } else {
    for (const m of models) {
      const c = m.capabilities;
      const flagsStr = [c.tools && "T", c.vision && "V", c.reasoning && "R"].filter(Boolean).join("") || "-";
      console.log(
        `${modelKey(m).padEnd(52)} [${flagsStr}] ${m.authRequired ? "key   " : "no-key"} ${m.name}`,
      );
    }
    console.log(`\n${models.length} model(s) across ${new Set(models.map((m) => m.namespace ?? m.provider)).size} vendor(s).`);
    if (client.warnings.length) console.log(`Warnings: ${JSON.stringify(client.warnings)}`);
  }
  process.exit(0);
}

if (cmd === "chat") {
  const message = String(opts["--message"] ?? rest.find((a) => !a.startsWith("--")) ?? "");
  if (!message) {
    console.error("No message given.");
    process.exit(2);
  }
  const chatOpts = {
    model: opts["--model"] ? String(opts["--model"]) : undefined,
    system: opts["--system"] ? String(opts["--system"]) : undefined,
    maxTokens: opts["--max-tokens"] ? Number(opts["--max-tokens"]) : undefined,
    temperature: opts["--temperature"] ? Number(opts["--temperature"]) : undefined,
  };

  if (opts["--stream"]) {
    for await (const c of client.stream(message, chatOpts)) {
      if (c.delta) process.stdout.write(c.delta);
      if (c.toolCall) console.error(`\n[tool call] ${JSON.stringify(c.toolCall)}`);
    }
    process.stdout.write("\n");
    process.exit(0);
  }

  const tools = opts["--tools"]
    ? [{
        type: "function",
        function: { name: "get_time", description: "Return the current time", parameters: { type: "object", properties: {} } },
      }]
    : undefined;

  const r = await client.chat(message, { ...chatOpts, tools });
  if (opts["--json"]) {
    console.log(JSON.stringify({ ...r, costUsd: 0 }, null, 2));
  } else {
    console.log(r.content);
    for (const tc of r.toolCalls) console.error(`[tool call] ${tc.name}(${JSON.stringify(tc.arguments)})`);
    console.error(`\n— ${r.modelId} · ${Math.round(r.latencyMs)} ms · ${r.usage.totalTokens} tok · $0.0000`);
  }
  process.exit(0);
}

if (cmd === "check") {
  console.log(JSON.stringify(client.status(), null, 2));
  process.exit(0);
}

console.error(`Unknown command: ${cmd ?? "(none)"}. Try list, chat, check.`);
process.exit(2);
