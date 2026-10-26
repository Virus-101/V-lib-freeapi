/** Base class for every provider. The contract is narrow on purpose. */

import type { ChatChunk, ChatResponse, Message, ModelInfo } from "../models.js";
import type { Transport } from "../transport.js";

export interface ChatArgs {
  messages: Message[];
  tools?: Record<string, unknown>[];
  temperature?: number;
  maxTokens?: number;
  system?: string;
  jsonMode?: boolean;
  seed?: number | null;
}

export abstract class BaseProvider {
  abstract readonly provider: string;
  protected constructor(
    readonly model: ModelInfo,
    readonly transport: Transport,
  ) {}

  abstract chat(args: ChatArgs): Promise<ChatResponse>;

  /**
   * Default streaming: one blocking call, one final chunk. Providers with real
   * streaming override this. The fallback exists so a consumer written against
   * `stream()` works on every provider.
   */
  async *stream(args: ChatArgs): AsyncGenerator<ChatChunk> {
    const resp = await this.chat(args);
    if (resp.reasoning) {
      yield { delta: "", reasoning: resp.reasoning, toolCall: null, finishReason: "", usage: null, modelId: resp.modelId, provider: this.provider as never };
    }
    yield { delta: resp.content, reasoning: "", toolCall: null, finishReason: resp.finishReason, usage: resp.usage, modelId: resp.modelId, provider: this.provider as never };
    for (const tc of resp.toolCalls) {
      yield { delta: "", reasoning: "", toolCall: { name: tc.name, arguments: tc.arguments }, finishReason: "", usage: null, modelId: resp.modelId, provider: this.provider as never };
    }
  }

  /** Cheapest possible reachability probe. False means unusable right now. */
  async healthCheck(): Promise<boolean> {
    try {
      await this.chat({ messages: [{ role: "user", content: "hi" }], maxTokens: 4 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Prepend a system message without mutating the caller's list.
   *
   * A bare string is coerced to a single user turn. The client does this, but
   * providers are reachable directly through `registry.get()`, and without the
   * coercion `["h","i"]` silently replaces one message with two one-character
   * ones — a request that looks valid and returns nonsense.
   */
  protected mergeSystem(messages: Message[], system?: string): Message[] {
    const base = typeof messages === "string" ? [{ role: "user", content: messages } as Message] : [...messages];
    if (system) base.unshift({ role: "system", content: system });
    return base;
  }
}
