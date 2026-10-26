/**
 * HTTP transport with retry, rate-limit parsing and SSE streaming.
 *
 * Two statuses mean different things and the caller must know which:
 *
 * - `QUOTA_STATUS` (402/429) — the caller hit a limit and must back off.
 * - `RETRYABLE_STATUS` — the server is having a bad moment; retry, then surface.
 *
 * After retries are exhausted a 500 must become a `ProviderError`, not a
 * `RateLimitError`. Routing a persistent 500 into the rate-limit branch sends
 * the caller off to sleep and retry something that will never succeed.
 */

import { AuthenticationError, ProviderError, RateLimitError } from "./errors.js";

const RETRYABLE = new Set([402, 408, 425, 429, 500, 502, 503, 504]);
const QUOTA = new Set([402, 429]);

export interface TransportOptions {
  timeoutMs?: number;
  maxRetries?: number;
  retryBackoffMs?: number;
  respectRetryAfter?: boolean;
  /** Injected for tests; defaults to global fetch. */
  fetchFn?: typeof fetch;
}

export interface RequestRecord {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

/** Test seam: record every request and replay canned responses. */
export interface MockTransport {
  requests: RequestRecord[];
  respond(input: string, init?: Record<string, unknown>): Promise<Record<string, unknown>>;
  streamLines(input: string, body: Record<string, unknown>, headers: Record<string, string>): AsyncGenerator<string>;
}

function retryAfterSeconds(headers: Headers, body: string): number | null {
  const ra = headers.get("retry-after") ?? headers.get("x-ratelimit-reset");
  if (ra) {
    const n = Number(ra);
    if (Number.isFinite(n)) return Math.max(0, n);
  }
  // Pollinations does not send Retry-After; it puts "Retry after 15s" in the body.
  const m = /retry after (\d+)/i.exec(body);
  return m ? Number(m[1]) : null;
}

function quotaOrProvider(status: number, url: string, body: string, headers: Headers): Error {
  if (QUOTA.has(status)) {
    return new RateLimitError(`${status} from ${url}: ${body.slice(0, 300)}`, retryAfterSeconds(headers, body));
  }
  return new ProviderError(`${status} from ${url}: ${body.slice(0, 300)}`, status, body);
}

export class Transport {
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly retryBackoffMs: number;
  private readonly respectRetryAfter: boolean;
  private readonly fetchFn: typeof fetch;

  constructor(opts: TransportOptions = {}) {
    this.timeoutMs = opts.timeoutMs ?? 300_000;
    this.maxRetries = opts.maxRetries ?? 3;
    this.retryBackoffMs = opts.retryBackoffMs ?? 1000;
    this.respectRetryAfter = opts.respectRetryAfter ?? true;
    this.fetchFn = opts.fetchFn ?? globalThis.fetch;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((r) => setTimeout(r, ms));
  }

  async request(
    method: string,
    url: string,
    body?: Record<string, unknown>,
    headers?: Record<string, string>,
  ): Promise<Response> {
    let attempt = 0;
    for (;;) {
      attempt += 1;
      let resp: Response;
      try {
        resp = await this.fetchFn(url, {
          method,
          headers: headers ?? { "content-type": "application/json" },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: AbortSignal.timeout(this.timeoutMs),
        });
      } catch (err) {
        if (attempt > this.maxRetries) {
          throw new ProviderError(`Request to ${url} failed: ${(err as Error).message}`);
        }
        await this.sleep(this.retryBackoffMs * 2 ** (attempt - 1));
        continue;
      }

      if (resp.status < 400) return resp;

      if (resp.status === 401 || resp.status === 403) {
        throw new AuthenticationError(`${resp.status} from ${url}: ${(await resp.text()).slice(0, 300)}`);
      }

      const text = await resp.text();

      if (RETRYABLE.has(resp.status) && attempt <= this.maxRetries) {
        const delay = this.respectRetryAfter ? retryAfterSeconds(resp.headers, text) : null;
        await this.sleep(delay ?? this.retryBackoffMs * 2 ** (attempt - 1));
        continue;
      }

      if (RETRYABLE.has(resp.status)) throw quotaOrProvider(resp.status, url, text, resp.headers);

      throw new ProviderError(`${resp.status} from ${url}: ${text.slice(0, 500)}`, resp.status, text);
    }
  }

  async postJson(url: string, body: Record<string, unknown>, headers?: Record<string, string>): Promise<Record<string, unknown>> {
    const resp = await this.request("POST", url, body, headers);
    const text = await resp.text();
    try {
      return JSON.parse(text) as Record<string, unknown>;
    } catch {
      throw new ProviderError(`Non-JSON response from ${url}: ${text.slice(0, 300)}`, resp.status);
    }
  }

  async getJson(url: string, headers?: Record<string, string>): Promise<unknown> {
    const resp = await this.request("GET", url, undefined, headers);
    const text = await resp.text();
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw new ProviderError(`Non-JSON response from ${url}: ${text.slice(0, 300)}`, resp.status);
    }
  }

  /** Yield complete response lines as they arrive; the caller parses the SSE framing. */
  async *streamLines(
    url: string,
    body: Record<string, unknown>,
    headers?: Record<string, string>,
  ): AsyncGenerator<string> {
    let attempt = 0;
    for (;;) {
      attempt += 1;
      const resp = await this.fetchFn(url, {
        method: "POST",
        headers: headers ?? { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (resp.status >= 400) {
        const text = await resp.text();
        if (RETRYABLE.has(resp.status) && attempt <= this.maxRetries) {
          const delay = this.respectRetryAfter ? retryAfterSeconds(resp.headers, text) : null;
          await this.sleep(delay ?? this.retryBackoffMs * 2 ** (attempt - 1));
          continue;
        }
        if (resp.status === 401 || resp.status === 403) {
          throw new AuthenticationError(`${resp.status} from ${url}: ${text.slice(0, 300)}`);
        }
        if (RETRYABLE.has(resp.status)) throw quotaOrProvider(resp.status, url, text, resp.headers);
        throw new ProviderError(`${resp.status} from ${url}: ${text.slice(0, 500)}`, resp.status, text);
      }

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      // Buffer across reads. A decoded chunk can split an SSE frame in half, and
      // yielding raw fragments means `data:` frames never parse — the caller sees
      // no text at all and nothing errors.
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          yield line.endsWith("\r") ? line.slice(0, -1) : line;
        }
      }
      // Flush any trailing partial line rather than dropping it.
      if (buf.length) yield buf;
      return;
    }
  }
}
