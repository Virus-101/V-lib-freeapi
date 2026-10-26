/**
 * Generic OpenAI-compatible provider.
 *
 * Nearly every LLM vendor speaks the same wire protocol — `POST
 * {baseUrl}/chat/completions` with an `Authorization: Bearer` header, plus `GET
 * {baseUrl}/models` for discovery. Hardcoding a class per vendor is busy work
 * that produces identical code fifteen times.
 *
 * So: one provider, driven by a `ProviderSpec`. Adding a free tier is a config
 * entry, not a library change.
 */

import { ConfigurationError, ProviderError } from "../errors.js";
import { Provider, capabilities, toolCallFromOpenAI, usageFromOpenAI } from "../models.js";
import type { Capabilities, ChatChunk, ChatResponse, Message, ModelInfo, Usage } from "../models.js";
import type { ProviderSpec } from "../presets.js";
import type { ChatArgs } from "./base.js";
import { BaseProvider } from "./base.js";

const DEFAULT_HEADERS: Record<string, string> = {
  "content-type": "application/json",
  "x-title": "freeaipi",
  "http-referer": "https://github.com/Virus-101/V-lib-freeapi",
};

export class OpenAICompatProvider extends BaseProvider {
  readonly provider = Provider.Compat;
  private readonly spec: ProviderSpec;
  private readonly apiKey: string;
  private readonly baseUrl: string;

  constructor(model: ModelInfo, transport: import("../transport.js").Transport, spec: ProviderSpec, apiKey: string) {
    super(model, transport);
    // Only demand a key when the spec declares one. A private vLLM or a local
    // endpoint legitimately needs no auth, and refusing it here would make the
    // generic provider unusable for exactly the case it is for.
    if (spec.keyEnv && !apiKey) {
      throw new ConfigurationError(`${spec.namespace} requires an API key (${spec.keyEnv}).`);
    }
    this.spec = spec;
    this.apiKey = apiKey;
    this.baseUrl = spec.baseUrl.replace(/\/$/, "");
  }

  private headers(): Record<string, string> {
    const h = { ...DEFAULT_HEADERS };
    if (this.apiKey) h.authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  private url(): string {
    return `${this.baseUrl}/chat/completions`;
  }

  private payload(args: ChatArgs, stream = false): Record<string, unknown> {
    const p: Record<string, unknown> = {
      model: this.model.providerModelId,
      messages: this.mergeSystem(args.messages as Message[], args.system),
      temperature: args.temperature ?? 0.7,
      max_tokens: args.maxTokens ?? 2048,
    };
    if (args.tools?.length) {
      p.tools = args.tools;
      p.tool_choice = "auto";
    }
    if (args.jsonMode) p.response_format = { type: "json_object" };
    if (args.seed !== null && args.seed !== undefined) p.seed = args.seed;
    if (stream) p.stream = true;
    return p;
  }

  async chat(args: ChatArgs): Promise<ChatResponse> {
    const started = Date.now();
    const data = await this.transport.postJson(this.url(), this.payload(args), this.headers());
    return this.toResponse(data, started);
  }

  private toResponse(data: Record<string, unknown>, started: number): ChatResponse {
    const err = data.error as Record<string, unknown> | undefined;
    if (err) throw new ProviderError(`${this.model.providerModelId} error: ${JSON.stringify(err)}`);
    const choices = (data.choices as Array<Record<string, unknown>> | undefined) ?? [];
    if (!choices.length) {
      throw new ProviderError(`${this.model.providerModelId} returned no choices: ${JSON.stringify(data).slice(0, 200)}`);
    }
    const choice = choices[0]!;
    const msg = (choice.message ?? {}) as Record<string, unknown>;
    const content = flatten(msg.content);
    return {
      content,
      modelId: this.model.modelId,
      provider: this.provider,
      usage: usageFromOpenAI(data.usage as Record<string, unknown> | undefined),
      toolCalls: ((msg.tool_calls as Array<Record<string, unknown>>) ?? []).map(toolCallFromOpenAI),
      finishReason: (choice.finish_reason as string) ?? "stop",
      latencyMs: Date.now() - started,
      reasoning: (msg.reasoning as string) ?? "",
      tier: this.model.free ? "free" : "",
      raw: data,
    };
  }

  async *stream(args: ChatArgs): AsyncGenerator<ChatChunk> {
    const started = Date.now();
    const pending = new Map<number, { name: string; arguments: string }>();
    let usage: Record<string, unknown> = {};
    const modelId = this.model.modelId;

    for await (const line of this.transport.streamLines(this.url(), this.payload(args, true), this.headers())) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") break;
      let obj: Record<string, unknown>;
      try {
        obj = JSON.parse(payload) as Record<string, unknown>;
      } catch {
        continue;
      }
      if (obj.usage) usage = obj.usage as Record<string, unknown>;
      const choices = (obj.choices as Array<Record<string, unknown>> | undefined) ?? [];
      if (!choices.length) continue; // usage-only frames are normal
      const delta = (choices[0]!.delta ?? {}) as Record<string, unknown>;
      if (typeof delta.content === "string" && delta.content) {
        yield { delta: delta.content, reasoning: "", toolCall: null, finishReason: "", usage: null, modelId, provider: this.provider };
      }
      if (typeof delta.reasoning === "string" && delta.reasoning) {
        yield { delta: "", reasoning: delta.reasoning, toolCall: null, finishReason: "", usage: null, modelId, provider: this.provider };
      }
      for (const tc of (delta.tool_calls as Array<Record<string, unknown>>) ?? []) {
        const idx = Number(tc.index ?? 0);
        const slot = pending.get(idx) ?? { name: "", arguments: "" };
        const fn = (tc.function ?? {}) as Record<string, unknown>;
        if (typeof fn.name === "string" && fn.name) slot.name = fn.name;
        if (typeof fn.arguments === "string" && fn.arguments) slot.arguments += fn.arguments;
        pending.set(idx, slot);
      }
      const finish = choices[0]!.finish_reason as string | undefined;
      if (finish) {
        for (const slot of pending.values()) {
          if (slot.name) {
            yield { delta: "", reasoning: "", toolCall: { name: slot.name, arguments: safeJson(slot.arguments) }, finishReason: finish, usage: null, modelId, provider: this.provider };
          }
        }
        pending.clear();
        yield { delta: "", reasoning: "", toolCall: null, finishReason: finish, usage: usageFromOpenAI(usage), modelId, provider: this.provider };
      }
    }
  }
}

export function flatten(c: unknown): string {
  if (!c) return "";
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((p) => (typeof p === "object" && p && typeof (p as { text?: unknown }).text === "string" ? ((p as { text: string }).text) : String(p ?? ""))).join("");
  return String(c);
}

export function safeJson(text: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(text || "{}");
    // `typeof [] === "object"`, so an array argument string must be excluded
    // explicitly — otherwise it is returned as an array rather than as _raw.
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : { _raw: text };
  } catch {
    return { _raw: text };
  }
}

// ── catalogue parsing ────────────────────────────────────────────────────────

export function parseCatalogue(raw: unknown, spec: ProviderSpec): ModelInfo[] {
  const entries: unknown[] = Array.isArray(raw) ? raw : ((raw as { data?: unknown[]; models?: unknown[] })?.data ?? (raw as { models?: unknown[] })?.models ?? []);
  const out: ModelInfo[] = [];
  for (const e of entries) {
    if (typeof e !== "object" || e === null) continue;
    const rec = e as Record<string, unknown>;
    const mid = (rec.id ?? rec.name ?? rec.slug) as string | undefined;
    if (!mid) continue;
    const ctx = Number(rec.context_length ?? ((rec.top_provider as Record<string, unknown>)?.context_length) ?? 32768);
    const maxOut = Number(rec.max_completion_tokens ?? ((rec.top_provider as Record<string, unknown>)?.max_completion_tokens) ?? Math.min(8192, Math.floor(ctx / 4)));
    out.push({
      modelId: mid,
      provider: Provider.Compat,
      providerModelId: mid,
      name: (rec.name as string) ?? mid,
      capabilities: capsFrom(rec, ctx, maxOut),
      free: spec.free,
      authRequired: true,
      rateLimit: spec.rateLimit,
      notes: spec.notes,
      namespace: spec.namespace,
    });
  }
  return out;
}

function capsFrom(rec: Record<string, unknown>, ctx: number, maxOut: number): Capabilities {
  const arch = (rec.architecture ?? {}) as Record<string, unknown>;
  const inMods = (arch.input_modalities ?? rec.input_modalities ?? ["text"]) as string[];
  const supported = (rec.supported_parameters as string[]) ?? [];
  const name = String(rec.id ?? rec.name ?? "").toLowerCase();
  return capabilities({
    text: true,
    vision: inMods.includes("image") || name.includes("vl") || name.includes("vision") || name.includes("llava"),
    tools: supported.includes("tools") || name.includes("tool") || name.includes("instruct") || name.includes("coder"),
    reasoning: supported.includes("reasoning") || name.includes("reason") || name.includes("r1") || name.includes("qwq"),
    jsonMode: supported.includes("response_format") || supported.includes("json"),
    contextLength: ctx,
    maxOutputTokens: maxOut,
  });
}

export async function fetchCatalogue(
  transport: import("../transport.js").Transport,
  spec: ProviderSpec,
  apiKey = "",
): Promise<ModelInfo[]> {
  const headers: Record<string, string> = { ...DEFAULT_HEADERS };
  if (apiKey) headers.authorization = `Bearer ${apiKey}`;
  try {
    const raw = await transport.getJson(spec.modelsUrl ?? `${spec.baseUrl.replace(/\/$/, "")}/models`, headers);
    return parseCatalogue(raw, spec);
  } catch {
    // A provider whose catalogue endpoint is down should leave the rest of the
    // registry intact, and the caller records the failure.
    return [];
  }
}
