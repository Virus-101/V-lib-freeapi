/**
 * The registry: which models exist, and which provider instance serves them.
 *
 * Built at client construction and immutable afterwards. It merges four sources:
 * Pollinations (no key), OpenRouter `:free` (key), local servers, and any
 * OpenAI-compatible endpoint enabled by an env var.
 *
 * Discovery failures are non-fatal and recorded in `warnings`: a network outage
 * must not make the library unusable, but it must be visible.
 */

import { ConfigurationError, ModelNotFoundError } from "./errors.js";
import { Provider, key as modelKey, label as modelLabel } from "./models.js";
import type { ModelInfo } from "./models.js";
import { readFileSync } from "node:fs";
import type { ChatArgs, BaseProvider } from "./providers/base.js";
import type { Transport } from "./transport.js";
import { OpenAICompatProvider, fetchCatalogue } from "./providers/openai_compat.js";
import { LocalProvider, buildModels, discover, inferCaps } from "./providers/local.js";
import { OpenRouterProvider, fetchFreeModels } from "./providers/openrouter.js";
import { PollinationsProvider, buildModels as plBuildModels } from "./providers/pollinations.js";
import { configPaths, enabledSpecs, loadConfigFile } from "./presets.js";
import type { ProviderSpec } from "./presets.js";

/** The slice of a provider the client actually calls. */
export interface ProviderLike {
  provider: Provider;
  chat(args: ChatArgs): Promise<unknown>;
  stream(args: ChatArgs): AsyncGenerator<unknown>;
}

export interface RegistryOptions {
  openrouterKey?: string;
  pollinationsKey?: string;
  localUrls?: string[];
  includeLocal?: boolean;
  includeOpenrouter?: boolean;
  includePollinations?: boolean;
  includeCompat?: boolean;
  compatSpecs?: ProviderSpec[];
  /** Injectable for tests. */
  readFile?: (p: string) => string;
}

export class Registry {
  readonly models = new Map<string, ModelInfo>();
  readonly providers = new Map<string, ProviderLike>();
  readonly warnings: string[] = [];

  get(key: string): { model: ModelInfo } {
    if (this.models.has(key)) return { model: this.models.get(key)! };
    const bare = key.split("/").pop()!;
    const matches = [...this.models.keys()].filter((k) => k.split("/").pop() === bare);
    if (matches.length === 1) return { model: this.models.get(matches[0]!)! };
    if (matches.length > 1) {
      throw new ModelNotFoundError(`'${key}' is ambiguous; use one of: ${[...matches].sort().join(", ")}`);
    }
    throw new ModelNotFoundError(
      `Model '${key}' not in registry. Known: ${[...this.models.keys()].sort().join(", ") || "(none)"}`,
    );
  }

  has(key: string): boolean {
    return this.models.has(key);
  }

  get size(): number {
    return this.models.size;
  }

  register(model: ModelInfo, provider: ProviderLike): void {
    const k = modelKey(model);
    this.models.set(k, model);
    this.providers.set(k, provider);
  }

  list(filters: {
    provider?: Provider | string;
    namespace?: string;
    requiresTools?: boolean;
    requiresVision?: boolean;
    requiresReasoning?: boolean;
    authRequired?: boolean;
    prefix?: string;
  } = {}): ModelInfo[] {
    // Coerce before the loop. Done inside it, an unknown provider kind is never
    // validated when the registry is empty — the loop body does not run and the
    // caller gets an empty list instead of an error naming the valid kinds.
    const want =
      filters.provider === undefined
        ? undefined
        : typeof filters.provider === "string"
          ? coerceProvider(filters.provider)
          : filters.provider;

    const out: ModelInfo[] = [];
    for (const m of this.models.values()) {
      if (filters.prefix && !modelKey(m).startsWith(filters.prefix)) continue;
      if (filters.namespace !== undefined && modelLabel(m) !== filters.namespace) continue;
      if (want !== undefined && m.provider !== want) continue;
      if (filters.requiresTools !== undefined && m.capabilities.tools !== filters.requiresTools) continue;
      if (filters.requiresVision !== undefined && m.capabilities.vision !== filters.requiresVision) continue;
      if (filters.requiresReasoning !== undefined && m.capabilities.reasoning !== filters.requiresReasoning) continue;
      if (filters.authRequired !== undefined && m.authRequired !== filters.authRequired) continue;
      out.push(m);
    }
    return out;
  }

  byProvider(): Map<string, ModelInfo[]> {
    const out = new Map<string, ModelInfo[]>();
    for (const m of this.models.values()) {
      const l = modelLabel(m);
      out.set(l, [...(out.get(l) ?? []), m]);
    }
    return out;
  }

  providersInUse(): string[] {
    return [...new Set([...this.models.values()].map(modelLabel))].sort();
  }
}

export function coerceProvider(value: Provider | string): Provider {
  if (typeof value !== "string") return value;
  const v = Object.values(Provider).find((p) => p === value);
  if (!v) {
    throw new ModelNotFoundError(
      `Unknown provider kind '${value}'. Known: ${Object.values(Provider).join(", ")}`,
    );
  }
  return v;
}

export async function createDefaultRegistry(transport: Transport, opts: RegistryOptions = {}): Promise<Registry> {
  const reg = new Registry();
  const readFile = opts.readFile ?? ((p: string) => readFileSync(p, "utf8"));

  if (opts.includePollinations !== false) {
    try {
      for (const m of await plBuildModels(transport)) {
        reg.register(m, new PollinationsProvider(m, transport, opts.pollinationsKey ?? ""));
      }
    } catch (e) {
      reg.warnings.push(`pollinations discovery failed: ${(e as Error).message}`);
    }
  }

  if (opts.includeOpenrouter) {
    try {
      const catalogue = await fetchFreeModels(transport, opts.openrouterKey ?? "");
      if (catalogue.length && !opts.openrouterKey) {
        // One line, not one per model: 17 identical messages read as a broken
        // library rather than a missing key.
        reg.warnings.push(
          `openrouter: ${catalogue.length} free models known but unavailable — set OPENROUTER_API_KEY to use them`,
        );
      } else if (catalogue.length) {
        for (const m of catalogue) {
          reg.register(m, new OpenRouterProvider(m, transport, opts.openrouterKey ?? ""));
        }
      }
    } catch (e) {
      reg.warnings.push(`openrouter discovery failed: ${(e as Error).message}`);
    }
  }

  if (opts.includeLocal !== false) {
    try {
      for (const target of await discover(transport, opts.localUrls)) {
        for (const m of buildModels(target.baseUrl, target.kind, target.models)) {
          reg.register(m, new LocalProvider(m, transport, target.baseUrl, target.kind));
        }
      }
    } catch (e) {
      reg.warnings.push(`local discovery failed: ${(e as Error).message}`);
    }
  }

  if (opts.includeCompat !== false) {
    let specs = opts.compatSpecs ?? enabledSpecs();
    for (const p of configPaths()) {
      try {
        specs = specs.concat(loadConfigFile(p, readFile));
      } catch (e) {
        reg.warnings.push(`provider config ${p} failed to load: ${(e as Error).message}`);
      }
    }

    for (const spec of specs) {
      try {
        // Inside the try on purpose: a provider whose key is missing or whose
        // catalogue is unreachable must leave the rest of the registry intact.
        const key = resolveKey(spec);
        let entries = await fetchCatalogue(transport, spec, key);
        if (spec.models?.length) {
          const filtered = entries.filter((m) => spec.models!.includes(m.modelId));
          entries = filtered.length ? filtered : spec.models!.map((id) => specOnlyModel(spec, id));
        }
        if (spec.exclude?.length) {
          entries = entries.filter((m) => !spec.exclude!.includes(m.modelId));
        }
        if (!entries.length) {
          reg.warnings.push(`${spec.namespace}: no models discovered (key present but catalogue empty or unreachable)`);
        }
        for (const m of entries) {
          reg.register(m, new OpenAICompatProvider(m, transport, spec, key));
        }
      } catch (e) {
        reg.warnings.push(`${spec.namespace} discovery failed: ${(e as Error).message}`);
      }
    }
  }

  return reg;
}

function resolveKey(spec: ProviderSpec): string {
  if (!spec.keyEnv) return ""; // a public endpoint (private vLLM) needs no auth
  const v = process.env[spec.keyEnv]?.trim() ?? "";
  if (!v) {
    throw new ConfigurationError(`${spec.namespace} is enabled but ${spec.keyEnv} is not set.`);
  }
  return v;
}

function specOnlyModel(spec: ProviderSpec, modelId: string): ModelInfo {
  return {
    modelId,
    provider: Provider.Compat,
    providerModelId: modelId,
    name: modelId,
    capabilities: inferCaps(modelId),
    free: spec.free,
    authRequired: Boolean(spec.keyEnv),
    rateLimit: spec.rateLimit,
    notes: spec.notes,
    namespace: spec.namespace,
  };
}
