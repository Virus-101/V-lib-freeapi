/**
 * The public client.
 *
 * One object holds credentials, the registry and the transport, and offers
 * `chat` / `stream` / `complete`. Routing is explicit by default (you name the
 * model) with an optional capability-based selector.
 */

import { ModelNotFoundError, RateLimitError } from "./errors.js";
import { Provider, label as modelLabel } from "./models.js";
import type { ChatChunk, ChatResponse, Message, ModelInfo } from "./models.js";
import { Transport } from "./transport.js";
import { createDefaultRegistry, Registry } from "./registry.js";
import type { RegistryOptions } from "./registry.js";

export interface ChatOptions {
  model?: string;
  system?: string;
  temperature?: number;
  maxTokens?: number;
  tools?: Record<string, unknown>[];
  jsonMode?: boolean;
  seed?: number | null;
  /** Model keys to try if the first fails. A quota error never falls back. */
  fallback?: string[];
}

export interface ClientOptions extends RegistryOptions {
  maxRetries?: number;
  /** Injectable for tests. */
  transport?: Transport;
  /** Injectable for tests: supply a registry and skip discovery entirely. */
  registry?: Registry;
}

const KNOWN_OPTIONS = new Set([
  "model", "system", "temperature", "maxTokens", "tools", "jsonMode", "seed", "fallback",
]);

export class FreeAIClient {
  readonly transport: Transport;
  readonly registry: Registry;
  private readonly orKey: string;
  private readonly plKey: string;

  constructor(opts: ClientOptions = {}) {
    this.transport = opts.transport ?? new Transport({ maxRetries: opts.maxRetries });
    this.orKey = opts.openrouterKey ?? process.env.OPENROUTER_API_KEY?.trim() ?? "";
    this.plKey = opts.pollinationsKey ?? process.env.POLLINATIONS_API_KEY?.trim() ?? "";
    if (opts.registry) {
      this.registry = opts.registry;
    } else {
      throw new Error("FreeAIClient must be constructed via FreeAIClient.create()");
    }
  }

  /** Async factory: construction performs network discovery, so it is awaited. */
  static async create(opts: ClientOptions = {}): Promise<FreeAIClient> {
    const transport = opts.transport ?? new Transport({ maxRetries: opts.maxRetries });
    const registry = await createDefaultRegistry(transport, {
      openrouterKey: opts.openrouterKey ?? process.env.OPENROUTER_API_KEY?.trim() ?? "",
      pollinationsKey: opts.pollinationsKey ?? process.env.POLLINATIONS_API_KEY?.trim() ?? "",
      localUrls: opts.localUrls,
      includeLocal: opts.includeLocal,
      includeOpenrouter: opts.includeOpenrouter,
      includePollinations: opts.includePollinations,
      includeCompat: opts.includeCompat,
      compatSpecs: opts.compatSpecs,
      readFile: opts.readFile,
    });
    return new FreeAIClient({ ...opts, transport, registry });
  }

  get warnings(): string[] {
    return this.registry.warnings;
  }

  // ── catalogue ───────────────────────────────────────────────────────────

  listModels(filters: Parameters<Registry["list"]>[0] = {}): ModelInfo[] {
    return this.registry.list(filters);
  }

  noKeyModels(): ModelInfo[] {
    return this.registry.list({ authRequired: false });
  }

  forVendor(namespace: string): ModelInfo[] {
    return this.registry.list({ namespace });
  }

  vendors(): string[] {
    return this.registry.providersInUse();
  }

  describe(key: string): ModelInfo {
    return this.registry.get(key).model;
  }

  // ── chat ────────────────────────────────────────────────────────────────

  async chat(messages: Message[] | string, opts: ChatOptions = {}): Promise<ChatResponse> {
    const args = this.buildArgs(messages, opts);
    const targets = this.targets(args, opts);
    if (!targets.length) {
      throw new ModelNotFoundError(`No model matched. Registry has: ${[...this.registry.models.keys()].sort().join(", ") || "(none)"}`);
    }
    for (const key of targets) {
      const provider = this.registry.providers.get(key);
      if (!provider) continue;
      try {
        return (await provider.chat(args)) as ChatResponse;
      } catch (e) {
        if (e instanceof RateLimitError) throw e; // quota is not a reason to silently switch models
        if (key !== targets[targets.length - 1]) continue;
        throw e;
      }
    }
    throw new ModelNotFoundError(`No provider available for: ${targets.join(", ")}`);
  }

  async *stream(messages: Message[] | string, opts: ChatOptions = {}): AsyncGenerator<ChatChunk> {
    const args = this.buildArgs(messages, opts);
    const targets = this.targets(args, opts);
    if (!targets.length) {
      throw new ModelNotFoundError(`No model matched. Registry has: ${[...this.registry.models.keys()].sort().join(", ") || "(none)"}`);
    }
    const provider = this.registry.providers.get(targets[0]!)!;
    for await (const c of provider.stream(args) as AsyncGenerator<ChatChunk>) yield c;
  }

  async complete(prompt: string, opts: ChatOptions = {}): Promise<string> {
    return (await this.chat(prompt, opts)).content;
  }

  // ── tool loop ───────────────────────────────────────────────────────────

  async runTools(
    messages: Message[],
    tools: Record<string, unknown>[],
    handlers: Record<string, (...a: unknown[]) => unknown>,
    opts: ChatOptions = {},
    maxRounds = 5,
  ): Promise<ChatResponse> {
    const msgs = [...messages];
    let round = 0;
    for (round = 1; round <= maxRounds; round++) {
      const resp = await this.chat(msgs, { ...opts, tools });
      if (!resp.toolCalls.length) {
        (resp.raw as Record<string, unknown>).rounds = round;
        return resp;
      }
      msgs.push({
        role: "assistant",
        content: resp.content || null,
        tool_calls: resp.toolCalls.map((tc) => ({
          id: tc.id,
          type: "function",
          function: { name: tc.name, arguments: tc.arguments },
        })),
      });
      for (const tc of resp.toolCalls) {
        const handler = handlers[tc.name];
        let result: string;
        try {
          result = handler ? String(await handler(tc.arguments)) : `no handler for ${tc.name}`;
        } catch (e) {
          result = `error: ${(e as Error).message}`;
        }
        msgs.push({ role: "tool", tool_call_id: tc.id, content: result });
      }
    }
    // Cap reached: one final call with tools removed so the model summarises.
    const resp = await this.chat(msgs, opts);
    (resp.raw as Record<string, unknown>).rounds = maxRounds;
    (resp.raw as Record<string, unknown>).round_cap_reached = true;
    return resp;
  }

  // ── routing ─────────────────────────────────────────────────────────────

  best(requirements: { requiresTools?: boolean; requiresVision?: boolean; requiresReasoning?: boolean; preferNoKey?: boolean } = {}): string {
    const candidates = this.registry.list({
      requiresTools: requirements.requiresTools,
      requiresVision: requirements.requiresVision,
      requiresReasoning: requirements.requiresReasoning,
    });
    if (!candidates.length) {
      throw new ModelNotFoundError(
        `No model matches those requirements. Try: ${this.registry.list().slice(0, 5).map(modelLabel).join(", ")}`,
      );
    }
    const preferNoKey = requirements.preferNoKey ?? true;
    candidates.sort(
      (a, b) =>
        Number(preferNoKey ? a.authRequired : false) - Number(preferNoKey ? b.authRequired : false) ||
        b.capabilities.contextLength - a.capabilities.contextLength,
    );
    return `${modelLabel(candidates[0]!)}/${candidates[0]!.modelId}`;
  }

  private buildArgs(messages: Message[] | string, opts: ChatOptions) {
    const extra = Object.keys(opts).filter((k) => !KNOWN_OPTIONS.has(k));
    if (extra.length) throw new TypeError(`Unknown chat option(s): ${extra.join(", ")}`);
    return {
      messages: typeof messages === "string" ? [{ role: "user", content: messages }] : messages,
      tools: opts.tools,
      temperature: opts.temperature,
      maxTokens: opts.maxTokens,
      system: opts.system,
      jsonMode: opts.jsonMode,
      seed: opts.seed,
    };
  }

  private targets(args: { tools?: Record<string, unknown>[] }, opts: ChatOptions): string[] {
    if (opts.model) return [opts.model, ...(opts.fallback ?? [])].filter((k): k is string => !!k);
    if (opts.fallback?.length) return opts.fallback;
    // Only filter on tools when tools were actually supplied. Passing
    // `requiresTools: false` for a plain text request would filter out every
    // tool-capable model.
    return [this.best({ requiresTools: args.tools?.length ? true : undefined })];
  }

  // ── introspection ───────────────────────────────────────────────────────

  status(): Record<string, unknown> {
    const redact = (k: string) => (k ? `<set len=${k.length} prefix=${k.slice(0, 4)}…>` : "<absent>");
    return {
      models: this.registry.size,
      providers: this.registry.providersInUse(),
      noKey: this.registry.list({ authRequired: false }).map((m) => `${modelLabel(m)}/${m.modelId}`),
      needsKey: this.registry.list({ authRequired: true }).map((m) => `${modelLabel(m)}/${m.modelId}`),
      credentials: {
        openrouter: redact(this.orKey),
        pollinations: redact(this.plKey),
      },
      warnings: this.registry.warnings,
    };
  }

  close(): void {
    // fetch keeps no connection pool to close; the hook exists for parity.
  }
}

export { Registry, Provider };
