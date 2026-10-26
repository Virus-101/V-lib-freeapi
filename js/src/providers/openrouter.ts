/**
 * OpenRouter's `:free` tier.
 *
 * `GET /api/v1/models` is public and returns the full catalogue with pricing;
 * the free models are those whose id ends in `:free` with zero pricing. `POST
 * /api/v1/chat/completions` requires an account key — the free tier is free
 * *money*, not free *account*.
 */

import { Provider, toolCallFromOpenAI, usageFromOpenAI } from "../models.js";
import type { ChatResponse, Message, ModelInfo } from "../models.js";
import type { ChatArgs } from "./base.js";
import { BaseProvider } from "./base.js";

export const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

const HEADERS: Record<string, string> = {
  "content-type": "application/json",
  "http-referer": "https://github.com/Virus-101/V-lib-freeapi",
  "x-title": "freeaipi",
};

export class OpenRouterProvider extends BaseProvider {
  readonly provider = Provider.OpenRouter;
  private readonly baseUrl: string;

  constructor(
    model: ModelInfo,
    transport: import("../transport.js").Transport,
    private readonly apiKey: string,
    baseUrl: string = OPENROUTER_BASE,
  ) {
    super(model, transport);
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private headers(): Record<string, string> {
    return { ...HEADERS, authorization: `Bearer ${this.apiKey}` };
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
    const data = await this.transport.postJson(`${this.baseUrl}/chat/completions`, this.payload(args), this.headers());
    if (data.error) throw new (await import("../errors.js")).ProviderError(`OpenRouter error: ${JSON.stringify(data.error)}`);
    const choices = (data.choices as Array<Record<string, unknown>>) ?? [];
    if (!choices.length) throw new (await import("../errors.js")).ProviderError(`OpenRouter returned no choices: ${JSON.stringify(data).slice(0, 200)}`);
    const choice = choices[0]!;
    const msg = (choice.message ?? {}) as Record<string, unknown>;
    return {
      content: flatten(msg.content),
      modelId: this.model.modelId,
      provider: this.provider,
      usage: usageFromOpenAI(data.usage as Record<string, unknown> | undefined),
      toolCalls: ((msg.tool_calls as Array<Record<string, unknown>>) ?? []).map(toolCallFromOpenAI),
      finishReason: (choice.finish_reason as string) ?? "stop",
      latencyMs: Date.now() - started,
      reasoning: (msg.reasoning as string) ?? "",
      tier: "free",
      raw: data,
    };
  }
}

function flatten(c: unknown): string {
  if (!c) return "";
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((p) => (typeof p === "object" && p && typeof (p as { text?: unknown }).text === "string" ? (p as { text: string }).text : String(p ?? ""))).join("");
  return String(c);
}

/** Fetch the `:free` tier. Empty array if the catalogue is unreachable. */
export async function fetchFreeModels(
  transport: import("../transport.js").Transport,
  apiKey = "",
): Promise<ModelInfo[]> {
  const headers: Record<string, string> = { ...HEADERS };
  if (apiKey) headers.authorization = `Bearer ${apiKey}`;
  let data: unknown;
  try {
    data = await transport.getJson(`${OPENROUTER_BASE}/models`, headers);
  } catch {
    return [];
  }
  const list = ((data as { data?: unknown[] })?.data ?? []) as Array<Record<string, unknown>>;
  const out: ModelInfo[] = [];
  for (const m of list) {
    const id = m.id as string | undefined;
    if (!id || !id.endsWith(":free")) continue;
    const pricing = (m.pricing ?? {}) as Record<string, unknown>;
    const paid = Number(pricing.prompt ?? 0) > 0 || Number(pricing.completion ?? 0) > 0;
    if (paid) continue;
    const arch = (m.architecture ?? {}) as Record<string, unknown>;
    const inMods = (arch.input_modalities ?? ["text"]) as string[];
    const supported = (m.supported_parameters as string[]) ?? [];
    const top = (m.top_provider ?? {}) as Record<string, unknown>;
    out.push({
      modelId: id,
      provider: Provider.OpenRouter,
      providerModelId: id,
      name: (m.name as string) ?? id,
      capabilities: {
        text: true,
        vision: inMods.includes("image"),
        tools: supported.includes("tools"),
        reasoning: supported.includes("reasoning") || supported.includes("include_reasoning"),
        jsonMode: supported.includes("response_format") || supported.includes("tools"),
        contextLength: Number(m.context_length ?? top.context_length ?? 32768),
        maxOutputTokens: Number(top.max_completion_tokens ?? 4096),
      },
      free: true,
      authRequired: true,
      rateLimit: "Free-tier quota per key; see GET /api/v1/key",
      notes: (m.description as string) ?? "",
    });
  }
  return out;
}
