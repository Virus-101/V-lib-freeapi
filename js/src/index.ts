/**
 * freeaipi — one interface for every free LLM API.
 *
 * Covers three kinds of zero-cost backend:
 *
 * 1. **Native free tiers** — services serving a free model with no account
 *    (Pollinations) or a free quota.
 * 2. **OpenRouter free models** — the `:free` tier behind a free account key.
 * 3. **Local** — Ollama and any OpenAI-compatible server on this machine.
 *
 * Nothing here charges money.
 */

export { FreeAIClient } from "./client.js";
export type { ChatOptions, ClientOptions } from "./client.js";
export { Registry, createDefaultRegistry, coerceProvider } from "./registry.js";
export type { RegistryOptions, ProviderLike } from "./registry.js";
export {
  Provider,
  capabilities,
  key,
  label,
  toolCallFromOpenAI,
  usageFromOpenAI,
} from "./models.js";
export type {
  Capabilities,
  ChatChunk,
  ChatResponse,
  Message,
  ModelInfo,
  ToolCall,
  Usage,
} from "./models.js";
export {
  FreeAIPIError,
  AuthenticationError,
  ConfigurationError,
  ModelNotFoundError,
  ProviderError,
  RateLimitError,
} from "./errors.js";
export { Transport } from "./transport.js";
export type { TransportOptions } from "./transport.js";
export { PRESETS, enabledSpecs, loadConfigFile, configPaths } from "./presets.js";
export type { ProviderSpec } from "./presets.js";
export {
  BaseProvider,
  PollinationsProvider,
  OpenRouterProvider,
  OpenAICompatProvider,
  LocalProvider,
  discover as discoverLocal,
} from "./providers/index.js";

import { FreeAIClient } from "./client.js";
import type { ChatChunk, ChatResponse, Message, ModelInfo } from "./models.js";
import type { ChatOptions, ClientOptions } from "./client.js";

export const VERSION = "0.1.0";

// ── one-call API ────────────────────────────────────────────────────────────
// The common case needs no client object. A single shared client is built on
// first use and reused, because construction performs network discovery.
// `reset()` rebuilds it — call after changing credentials.

let shared: FreeAIClient | null = null;

function client(): FreeAIClient {
  if (!shared) throw new Error("freeaipi: await freeaipi.init() before using the one-call API");
  return shared;
}

/** Build the shared client. Required before the one-call API. */
export async function init(opts: ClientOptions = {}): Promise<FreeAIClient> {
  shared = await FreeAIClient.create(opts);
  return shared;
}

/** Discard and rebuild the shared client (e.g. after changing credentials). */
export async function reset(opts: ClientOptions = {}): Promise<FreeAIClient> {
  shared = await FreeAIClient.create(opts);
  return shared;
}

export function chat(messages: Message[] | string, opts?: ChatOptions): Promise<ChatResponse> {
  return client().chat(messages, opts);
}

export function complete(prompt: string, opts?: ChatOptions): Promise<string> {
  return client().complete(prompt, opts);
}

export function stream(messages: Message[] | string, opts?: ChatOptions): AsyncGenerator<ChatChunk> {
  return client().stream(messages, opts);
}

export function listModels(filters?: Parameters<FreeAIClient["listModels"]>[0]): ModelInfo[] {
  return client().listModels(filters);
}

export function noKeyModels(): ModelInfo[] {
  return client().noKeyModels();
}

export function forVendor(namespace: string): ModelInfo[] {
  return client().forVendor(namespace);
}

export function vendors(): string[] {
  return client().vendors();
}

export function describe(key: string): ModelInfo {
  return client().describe(key);
}

export function best(requirements?: Parameters<FreeAIClient["best"]>[0]): string {
  return client().best(requirements);
}

export function status(): Record<string, unknown> {
  return client().status();
}

export function warnings(): string[] {
  return shared ? shared.warnings : [];
}
