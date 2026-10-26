/**
 * Curated specs for free-tier OpenAI-compatible endpoints.
 *
 * Adding a provider is adding an entry here — a class per vendor would mean
 * fifteen copies of identical request-shaping code.
 *
 * A provider is enabled automatically when its key env var is present. That is
 * what makes "call all the free APIs" practical: set a key, the provider
 * appears, no config file and no code change.
 *
 * Honesty about which are free
 * ----------------------------
 * `free: true` means the vendor publishes a no-cost tier. It does **not** mean
 * unlimited — `rateLimit` states the real constraint. Three entries are not
 * free at all and are included only because they are common: they are marked
 * `free: false` so the catalogue cannot be read as a promise.
 */

export interface ProviderSpec {
  /** Vendor name, used in the registry key. */
  namespace: string;
  baseUrl: string;
  /** Env var holding the credential. Empty string => no auth (private server). */
  keyEnv: string;
  free: boolean;
  rateLimit: string;
  notes: string;
  /** Whether the endpoint was actually called during development. */
  verified: boolean;
  /** Catalogue URL; defaults to `${baseUrl}/models`. */
  modelsUrl?: string;
  /** Only register these ids; empty means everything in the catalogue. */
  models?: string[];
  /** Skip these ids. */
  exclude?: string[];
}

export const PRESETS: Record<string, ProviderSpec> = {
  groq: {
    namespace: "groq",
    baseUrl: "https://api.groq.com/openai/v1",
    keyEnv: "GROQ_API_KEY",
    free: true,
    rateLimit: "Free tier: per-day request and token caps; resets daily",
    notes: "Fastest inference available; Llama, Qwen, Gemma, Mistral.",
    verified: true,
  },
  cerebras: {
    namespace: "cerebras",
    baseUrl: "https://api.cerebras.ai/v1",
    keyEnv: "CEREBRAS_API_KEY",
    free: true,
    rateLimit: "Free tier: per-hour and per-day caps",
    notes: "Very high throughput on Llama and Qwen weights.",
    verified: false,
  },
  sambanova: {
    namespace: "sambanova",
    baseUrl: "https://api.sambanova.ai/v1",
    keyEnv: "SAMBANOVA_API_KEY",
    free: true,
    rateLimit: "Free tier: per-day quota",
    notes: "DeepSeek-V3, Llama-3.3, MiniMax. Catalogue endpoint is public without a key.",
    verified: true,
  },
  together: {
    namespace: "together",
    baseUrl: "https://api.together.xyz/v1",
    keyEnv: "TOGETHER_API_KEY",
    free: true,
    rateLimit: "Free tier: credit grant on signup, then paid",
    notes: "The free grant is a one-off, not a standing tier.",
    verified: false,
  },
  fireworks: {
    namespace: "fireworks",
    baseUrl: "https://api.fireworks.ai/inference/v1",
    keyEnv: "FIREWORKS_API_KEY",
    free: true,
    rateLimit: "Free tier: per-day token grant",
    notes: "DeepSeek and Llama serving with function calling.",
    verified: false,
  },
  mistral: {
    namespace: "mistral",
    baseUrl: "https://api.mistral.ai/v1",
    keyEnv: "MISTRAL_API_KEY",
    free: true,
    rateLimit: "'Free' (Discovery) plan: per-day message cap, small models only",
    notes: "The free plan excludes the flagship models.",
    verified: false,
  },
  deepseek: {
    namespace: "deepseek",
    baseUrl: "https://api.deepseek.com/v1",
    keyEnv: "DEEPSEEK_API_KEY",
    free: false,
    rateLimit: "Paid; small signup credit only",
    notes: "NOT free. Included because it is a frequent question and its pricing is unusually low.",
    verified: true,
  },
  novita: {
    namespace: "novita",
    baseUrl: "https://api.novita.ai/v3/openai",
    keyEnv: "NOVITA_API_KEY",
    free: true,
    rateLimit: "Free tier: monthly token grant",
    notes: "DeepSeek, Llama, Qwen and others; occasional free promotions.",
    verified: true,
  },
  nous: {
    namespace: "nous",
    baseUrl: "https://inference-api.nousresearch.com/v1",
    keyEnv: "NOUS_API_KEY",
    free: false,
    rateLimit: "Paid; Hermes manages a rotating OAuth token",
    notes: "NOT free — Hermes uses it. Shown so it is not confused with the free tiers.",
    verified: true,
  },
  inception: {
    namespace: "inception",
    baseUrl: "https://api.inceptionlabs.ai/v1",
    keyEnv: "INCEPTION_API_KEY",
    free: false,
    rateLimit: "Paid; Mercury diffusion models",
    notes: "NOT free. Included for completeness.",
    verified: true,
  },
};

/** Presets whose key env var is present and non-empty. */
export function enabledSpecs(env: Record<string, string | undefined> = process.env): ProviderSpec[] {
  return Object.values(PRESETS).filter((s) => (env[s.keyEnv] ?? "").trim().length > 0);
}

/** Load extra providers from a JSON file. See the README for the shape. */
export function loadConfigFile(path: string, readFile: (p: string) => string): ProviderSpec[] {
  let raw: unknown;
  try {
    raw = JSON.parse(readFile(path));
  } catch {
    return [];
  }
  const list = (raw as { providers?: unknown[] })?.providers ?? [];
  const out: ProviderSpec[] = [];
  for (const entry of list) {
    const e = entry as Partial<ProviderSpec>;
    if (!e || typeof e.baseUrl !== "string") continue;
    out.push({
      namespace: e.namespace ?? "custom",
      baseUrl: e.baseUrl,
      keyEnv: e.keyEnv ?? "",
      free: e.free ?? true,
      rateLimit: e.rateLimit ?? "",
      notes: e.notes ?? "",
      verified: false,
      models: e.models,
      exclude: e.exclude,
    });
  }
  return out;
}

/** Env vars pointing at provider config files, in priority order. */
export function configPaths(env: Record<string, string | undefined> = process.env): string[] {
  return [env.FREEAIPI_PROVIDERS, env.FREEAIPI_PROVIDERS_JSON].filter((v): v is string => !!v && v.trim().length > 0);
}
