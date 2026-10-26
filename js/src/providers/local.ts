/**
 * Local backends — Ollama, llama.cpp, vLLM, LM Studio, Jan, GPT4All.
 *
 * All speak the OpenAI chat protocol on localhost, so they share one
 * implementation and differ only in default port. They are the only backends
 * with no network dependency, no account and no rate limit — and the only ones
 * that fail when a process is not running, which is why discovery is a probe
 * that reports "not running" instead of throwing.
 */

import { Provider, capabilities } from "../models.js";
import type { Capabilities, ChatResponse, Message, ModelInfo } from "../models.js";
import type { ChatArgs } from "./base.js";
import { BaseProvider } from "./base.js";

export const KNOWN_TARGETS: Array<{ name: string; baseUrl: string; kind: "ollama" | "openai" }> = [
  { name: "ollama", baseUrl: "http://127.0.0.1:11434", kind: "ollama" },
  { name: "llamacpp", baseUrl: "http://127.0.0.1:8080/v1", kind: "openai" },
  { name: "vllm", baseUrl: "http://127.0.0.1:8000/v1", kind: "openai" },
  { name: "lmstudio", baseUrl: "http://127.0.0.1:1234/v1", kind: "openai" },
  { name: "jan", baseUrl: "http://127.0.0.1:1337/v1", kind: "openai" },
  { name: "gpt4all", baseUrl: "http://127.0.0.1:4891/v1", kind: "openai" },
];

export interface LocalTarget {
  name: string;
  baseUrl: string;
  kind: "ollama" | "openai";
  models: string[];
}

/** Probe every known local endpoint. A dead port is skipped silently. */
export async function discover(
  transport: import("../transport.js").Transport,
  extraUrls: string[] = [],
  timeoutMs = 2000,
): Promise<LocalTarget[]> {
  const targets = [
    ...KNOWN_TARGETS,
    ...extraUrls.map((u) => ({ name: "custom", baseUrl: u.replace(/\/$/, ""), kind: "openai" as const })),
  ];
  const found: LocalTarget[] = [];
  for (const t of targets) {
    try {
      if (t.kind === "ollama") {
        const data = (await transport.getJson(`${t.baseUrl}/api/tags`)) as { models?: Array<{ name?: string }> };
        const names = (data?.models ?? []).map((m) => m.name).filter((n): n is string => !!n);
        if (names.length) found.push({ ...t, models: names });
      } else {
        const data = (await transport.getJson(`${t.baseUrl}/models`)) as { data?: Array<{ id?: string }> };
        const ids = (data?.data ?? []).map((m) => m.id).filter((n): n is string => !!n);
        if (ids.length) found.push({ ...t, models: ids });
      }
    } catch {
      continue;
    }
  }
  return found;
}

export class LocalProvider extends BaseProvider {
  readonly provider = Provider.Local;
  private readonly baseUrl: string;
  private readonly kind: "ollama" | "openai";

  constructor(model: ModelInfo, transport: import("../transport.js").Transport, baseUrl: string, kind: "ollama" | "openai" = "openai") {
    super(model, transport);
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.kind = kind;
  }

  async chat(args: ChatArgs): Promise<ChatResponse> {
    const started = Date.now();
    const msgs = this.mergeSystem(args.messages as Message[], args.system);
    const payload: Record<string, unknown> =
      this.kind === "ollama"
        ? {
            model: this.model.providerModelId,
            messages: msgs,
            stream: false,
            options: { temperature: args.temperature ?? 0.7, num_predict: args.maxTokens ?? 2048 },
          }
        : {
            model: this.model.providerModelId,
            messages: msgs,
            temperature: args.temperature ?? 0.7,
            max_tokens: args.maxTokens ?? 2048,
          };

    if (args.tools?.length) payload.tools = args.tools;
    if (args.seed !== null && args.seed !== undefined) {
      if (this.kind === "ollama") (payload.options as Record<string, unknown>).seed = args.seed;
      else payload.seed = args.seed;
    }
    if (args.jsonMode) payload.format = "json";

    const url = this.kind === "ollama" ? `${this.baseUrl}/api/chat` : `${this.baseUrl}/chat/completions`;
    const data = (await this.transport.postJson(url, payload)) as Record<string, unknown>;
    if (data.error) throw new (await import("../errors.js")).ProviderError(`${this.model.providerModelId} error: ${JSON.stringify(data.error)}`);

    if (this.kind === "ollama") {
      const msg = (data.message ?? {}) as Record<string, unknown>;
      const tcs = (msg.tool_calls as Array<Record<string, unknown>>) ?? [];
      const toolCalls = tcs.map((tc, i) => ({
        id: `call_${i}`,
        name: String(((tc.function as Record<string, unknown>) ?? {}).name ?? ""),
        arguments: ((tc.function as Record<string, unknown>) ?? {}).arguments as Record<string, unknown> ?? {},
      }));
      return {
        content: String(msg.content ?? ""),
        modelId: this.model.modelId,
        provider: this.provider,
        usage: {
          promptTokens: Number(data.prompt_eval_count ?? 0),
          completionTokens: Number(data.eval_count ?? 0),
          totalTokens: Number(data.prompt_eval_count ?? 0) + Number(data.eval_count ?? 0),
        },
        toolCalls,
        finishReason: String(data.done_reason ?? (data.done ? "stop" : "")),
        latencyMs: Date.now() - started,
        reasoning: "",
        tier: "local",
        raw: data,
      };
    }

    const choices = (data.choices as Array<Record<string, unknown>>) ?? [];
    if (!choices.length) throw new (await import("../errors.js")).ProviderError(`no choices from ${this.model.providerModelId}`);
    const choice = choices[0]!;
    const msg = (choice.message ?? {}) as Record<string, unknown>;
    return {
      content: String(msg.content ?? ""),
      modelId: this.model.modelId,
      provider: this.provider,
      usage: {
        promptTokens: Number((data.usage as Record<string, unknown>)?.prompt_tokens ?? 0),
        completionTokens: Number((data.usage as Record<string, unknown>)?.completion_tokens ?? 0),
        totalTokens: Number((data.usage as Record<string, unknown>)?.total_tokens ?? 0),
      },
      toolCalls: ((msg.tool_calls as Array<Record<string, unknown>>) ?? []).map((tc) => ({
        id: String(tc.id ?? ""),
        name: String(((tc.function as Record<string, unknown>) ?? {}).name ?? ""),
        arguments: ((((tc.function as Record<string, unknown>) ?? {}).arguments as string) || "{}")
          ? JSON.parse((((tc.function as Record<string, unknown>) ?? {}).arguments as string) || "{}")
          : {},
      })),
      finishReason: String(choice.finish_reason ?? "stop"),
      latencyMs: Date.now() - started,
      reasoning: "",
      tier: "local",
      raw: data,
    };
  }
}

const SIZE_CTX: Array<[number, number]> = [
  [100, 1_000_000],
  [512, 512_000],
  [256, 262_144],
  [128, 131_072],
  [64, 65_536],
  [32, 32_768],
  [16, 16_384],
  [8, 8_192],
  [4, 4_096],
];

/** Guess capabilities from a model name. A hint, not a contract. */
export function inferCaps(name: string): Capabilities {
  const lowered = name.toLowerCase();
  let ctx = 8192;
  for (const [marker, size] of SIZE_CTX) {
    if (lowered.includes(`${marker}k`)) {
      ctx = size;
      break;
    }
  }
  const segments = lowered.replace(/:/g, "-").split("-");
  const vision = lowered.includes("vision") || lowered.includes("llava") || segments.includes("vl");
  const tools = ["tool", "function", "agentic", "coder", "instruct"].some((k) => lowered.includes(k));
  const reasoning = ["reason", "think", "r1", "qwq", "deepseek"].some((k) => lowered.includes(k));
  return capabilities({
    text: true,
    vision,
    tools,
    reasoning,
    contextLength: ctx,
    maxOutputTokens: Math.min(8192, Math.floor(ctx / 2)),
  });
}

export function buildModels(baseUrl: string, kind: string, names: string[]): ModelInfo[] {
  return names.map((name) => {
    const bare = name.split(":")[0]!;
    return {
      modelId: bare,
      provider: Provider.Local,
      providerModelId: name,
      name: `${bare} (local)`,
      capabilities: inferCaps(bare),
      free: true,
      authRequired: false,
      rateLimit: "",
      notes: `${kind} @ ${baseUrl}`,
    };
  });
}
