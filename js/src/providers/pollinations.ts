/**
 * Pollinations Text API — the only zero-key hosted backend.
 *
 * Verified against the live endpoint on 2026-10-26:
 * - `POST https://text.pollinations.ai/openai` is OpenAI-shaped and works with
 *   **no key at all**. `POST /openai/` also works.
 * - The response carries `user_tier: "anonymous"` and real usage numbers.
 * - `reasoning` is a separate field on the message, alongside `content`.
 * - Tool calls come back in standard OpenAI `tool_calls` form.
 *
 * The one thing that will bite a user is the anonymous quota, enforced per IP:
 *
 *     HTTP 402 {"code": "FREE_LIMIT",
 *               "error": {"message": "Free limit reached: 1 reply every 30
 *                          seconds per IP. Retry after 15s, ..."}}
 *
 * A `referrer` in the payload was observed to lift that limit on the anonymous
 * tier, so it is sent by default. A pollen key takes precedence; with a key the
 * request goes out as a bearer token and the 402 disappears.
 */

import { ProviderError, RateLimitError } from "../errors.js";
import { Provider, toolCallFromOpenAI, usageFromOpenAI } from "../models.js";
import type { ChatChunk, ChatResponse, Message, ModelInfo } from "../models.js";
import type { ChatArgs } from "./base.js";
import { BaseProvider } from "./base.js";

export const POLLINATIONS_BASE = "https://text.pollinations.ai";
export const POLLINATIONS_REFERRER = "freeaipi";
export const POLLINATIONS_DEFAULT_MODEL = "openai-fast";

export interface PollinationsMessage {
  content?: string | Array<{ type?: string; text?: string }>;
  reasoning?: string;
  tool_calls?: Array<Record<string, unknown>>;
}

export function flattenContent(c: PollinationsMessage["content"]): string {
  if (!c) return "";
  if (typeof c === "string") return c;
  return c.map((p) => (typeof p === "object" && p && typeof p.text === "string" ? p.text : String(p ?? ""))).join("");
}

/** Parse a Pollinations-style SSE line into a partial chunk. */
export function parseSSELine(line: string): Record<string, unknown> | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data:")) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === "[DONE]") return null;
  try {
    return JSON.parse(payload) as Record<string, unknown>;
  } catch {
    return null; // a partial line at a chunk boundary is not fatal
  }
}

export function isQuotaError(body: string): number | null {
  const m = /retry after (\d+)/i.exec(body);
  return m ? Number(m[1]) : null;
}

export function quotaError(body: string): RateLimitError {
  return new RateLimitError(
    `Pollinations free quota reached: ${body.slice(0, 200)}`,
    isQuotaError(body),
  );
}

export const POLLINATIONS_HEADERS = { "content-type": "application/json", accept: "application/json" };

export function pollinationsUrl(baseUrl: string = POLLINATIONS_BASE): string {
  // `/openai` is the documented OpenAI-compatible path. `/` alone answers with
  // the GET-style model query, which is NOT the chat endpoint.
  return `${baseUrl.replace(/\/$/, "")}/openai`;
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

export class PollinationsProvider extends BaseProvider {
  readonly provider = Provider.Pollinations;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly referrer: string;

  constructor(
    model: ModelInfo,
    transport: import("../transport.js").Transport,
    apiKey = "",
    baseUrl: string = POLLINATIONS_BASE,
    referrer: string = POLLINATIONS_REFERRER,
  ) {
    super(model, transport);
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.referrer = referrer;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "content-type": "application/json", accept: "application/json" };
    if (this.apiKey) h.authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  private payload(args: ChatArgs, stream = false): Record<string, unknown> {
    const p: Record<string, unknown> = {
      model: this.model.providerModelId,
      messages: this.mergeSystem(args.messages as Message[], args.system),
      temperature: args.temperature ?? 0.7,
      max_tokens: args.maxTokens ?? 2048,
      referrer: this.referrer,
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
    const url = `${this.baseUrl}/openai`;

    // Pollinations reports the quota as a non-throwing HTTP 402 with a JSON
    // envelope, so the status is read directly rather than going through the
    // generic error path.
    const resp = await this.transport.request("POST", url, this.payload(args), this.headers());
    const text = await resp.text();

    if (resp.status === 402) throw quotaError(text);
    if (resp.status >= 400) throw new ProviderError(`Pollinations HTTP ${resp.status}: ${text.slice(0, 300)}`, resp.status);

    let data: Record<string, unknown>;
    try {
      data = JSON.parse(text) as Record<string, unknown>;
    } catch {
      throw new ProviderError(`Pollinations returned non-JSON: ${text.slice(0, 200)}`, resp.status);
    }
    return this.toResponse(data, started);
  }

  private toResponse(data: Record<string, unknown>, started: number): ChatResponse {
    const choices = (data.choices as Array<Record<string, unknown>> | undefined) ?? [];
    if (!choices.length) throw new ProviderError(`Pollinations returned no choices: ${JSON.stringify(data).slice(0, 200)}`);
    const choice = choices[0]!;
    const msg = (choice.message ?? {}) as PollinationsMessage;
    return {
      content: flattenContent(msg.content),
      modelId: this.model.modelId,
      provider: this.provider,
      usage: usageFromOpenAI(data.usage as Record<string, unknown> | undefined),
      toolCalls: (msg.tool_calls ?? []).map(toolCallFromOpenAI),
      finishReason: (choice.finish_reason as string) ?? "stop",
      latencyMs: Date.now() - started,
      reasoning: msg.reasoning ?? "",
      tier: String(data.user_tier ?? ""),
      raw: data,
    };
  }

  async *stream(args: ChatArgs): AsyncGenerator<ChatChunk> {
    const modelId = this.model.modelId;
    const pending = new Map<number, { name: string; arguments: string }>();
    let usage: Record<string, unknown> = {};

    for await (const line of this.transport.streamLines(`${this.baseUrl}/openai`, this.payload(args, true), this.headers())) {
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

/** Discover Pollinations' anonymous models. Falls back to the known entry. */
export async function buildModels(transport: import("../transport.js").Transport): Promise<ModelInfo[]> {
  const fallback: ModelInfo = {
    modelId: POLLINATIONS_DEFAULT_MODEL,
    provider: Provider.Pollinations,
    providerModelId: "gpt-oss-20b",
    name: "GPT-OSS 20B (Pollinations)",
    capabilities: { text: true, vision: false, tools: true, reasoning: true, jsonMode: true, contextLength: 131072, maxOutputTokens: 8192 },
    free: true,
    authRequired: false,
    rateLimit: "1 reply / 30 s per IP (anonymous); unlimited with a pollen key",
    notes: "No key required. Open-weight reasoning model.",
  };

  let raw: unknown;
  try {
    raw = await transport.getJson(`${POLLINATIONS_BASE}/models`, { accept: "application/json" });
  } catch {
    return [fallback];
  }

  const list = Array.isArray(raw) ? (raw as Array<Record<string, unknown>>) : [];
  const out: ModelInfo[] = [];
  for (const entry of list) {
    const name = entry.name as string | undefined;
    if (!name) continue;
    const caps = entry.input_modalities as string[] | undefined;
    const aliases = (entry.aliases as string[]) ?? [];
    out.push({
      modelId: name,
      provider: Provider.Pollinations,
      providerModelId: aliases[0] ?? name,
      name: `${name} (Pollinations)`,
      capabilities: {
        text: true,
        vision: (caps ?? []).includes("image"),
        tools: Boolean(entry.tools),
        reasoning: Boolean(entry.reasoning),
        jsonMode: Boolean(entry.tools),
        contextLength: Number(entry.context_length ?? 131072),
        maxOutputTokens: Number(entry.max_output_tokens ?? 8192),
      },
      free: true,
      authRequired: false,
      rateLimit: "1 reply / 30 s per IP (anonymous); unlimited with a pollen key",
      notes: String(entry.description ?? ""),
    });
  }
  return out.length ? out : [fallback];
}
