/**
 * Shared data types. Small on purpose: a provider converts its own wire format
 * into these, so the client and registry never learn what a vendor calls its
 * fields.
 */

/** A zero-cost backend kind. The *implementation* identity, not the vendor. */
export enum Provider {
  Pollinations = "pollinations",
  OpenRouter = "openrouter",
  Inception = "inception",
  Local = "local",
  /** Any OpenAI-compatible endpoint configured at runtime (groq, cerebras…). */
  Compat = "openai_compat",
}

export interface Capabilities {
  text: boolean;
  vision: boolean;
  tools: boolean;
  reasoning: boolean;
  jsonMode: boolean;
  contextLength: number;
  maxOutputTokens: number;
}

export function capabilities(partial: Partial<Capabilities> = {}): Capabilities {
  return {
    text: true,
    vision: false,
    tools: false,
    reasoning: false,
    jsonMode: false,
    contextLength: 8192,
    maxOutputTokens: 4096,
    ...partial,
  };
}

export interface ModelInfo {
  modelId: string;
  provider: Provider;
  /** The id the backend expects on the wire. May differ from `modelId`. */
  providerModelId: string;
  name: string;
  capabilities: Capabilities;
  free: boolean;
  authRequired: boolean;
  rateLimit: string;
  notes: string;
  /**
   * The vendor name used in the registry key. Defaults to the provider kind and
   * is overridden for generic endpoints so the key reads
   * `groq/llama-3.3-70b` rather than `openai_compat/llama-3.3-70b`.
   */
  namespace?: string;
}

export function label(m: ModelInfo): string {
  return m.namespace ?? m.provider;
}

export function key(m: ModelInfo): string {
  return `${label(m)}/${m.modelId}`;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export function toolCallFromOpenAI(raw: Record<string, unknown>): ToolCall {
  const fn = (raw.function ?? {}) as Record<string, unknown>;
  const argsRaw = (fn.arguments ?? "") as string;
  let args: Record<string, unknown>;
  if (argsRaw && typeof argsRaw === "object") {
    args = argsRaw as Record<string, unknown>;
  } else {
    try {
      const parsed = JSON.parse(argsRaw || "{}");
      args = typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : { _raw: argsRaw };
    } catch {
      // A malformed argument string must surface, not crash the turn.
      args = { _raw: argsRaw };
    }
  }
  return {
    id: (raw.id as string) ?? "",
    name: (fn.name as string) ?? "",
    arguments: args,
  };
}

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export function usageFromOpenAI(raw?: Record<string, unknown> | null): Usage {
  const u = raw ?? {};
  return {
    promptTokens: Number(u.prompt_tokens ?? 0),
    completionTokens: Number(u.completion_tokens ?? 0),
    totalTokens: Number(u.total_tokens ?? 0),
  };
}

export interface ChatResponse {
  content: string;
  modelId: string;
  provider: Provider;
  usage: Usage;
  toolCalls: ToolCall[];
  finishReason: string;
  latencyMs: number;
  /** Chain-of-thought, kept separate from `content` — never glued on. */
  reasoning: string;
  /** Provider's own tier marker, e.g. "anonymous" for Pollinations. */
  tier: string;
  raw: Record<string, unknown>;
}

export interface ChatChunk {
  /** Incremental text. Exactly one field is set per chunk. */
  delta: string;
  reasoning: string;
  toolCall: { name: string; arguments: Record<string, unknown> } | null;
  finishReason: string;
  usage: Usage | null;
  modelId: string;
  provider: Provider | null;
}

export function chunk(partial: Partial<ChatChunk> = {}): ChatChunk {
  return {
    delta: "",
    reasoning: "",
    toolCall: null,
    finishReason: "",
    usage: null,
    modelId: "",
    provider: null,
    ...partial,
  };
}

export type Message = Record<string, unknown>;
