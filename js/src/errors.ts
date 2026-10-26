/**
 * Error hierarchy for freeaipi.
 *
 * Every error the library throws extends `FreeAIPIError`, so a caller can catch
 * the family with one `catch`. The subclasses exist because the handling differs
 * in kind: a rate limit wants a retry after a delay, an auth error wants a
 * configuration change, a provider error wants a fallback model.
 */

export class FreeAIPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** A required credential or setting is missing. Raised before any request. */
export class ConfigurationError extends FreeAIPIError {}

/** The provider rejected the credential (HTTP 401/403). */
export class AuthenticationError extends FreeAIPIError {}

/** Quota refused the request (HTTP 402/429). Carries the retry hint. */
export class RateLimitError extends FreeAIPIError {
  /** Seconds to wait, when the provider published one. */
  readonly retryAfter: number | null;

  constructor(message: string, retryAfter: number | null = null) {
    super(message);
    this.retryAfter = retryAfter;
  }
}

/** A non-success response that is not auth or quota. */
export class ProviderError extends FreeAIPIError {
  readonly statusCode: number | null;
  readonly body: string | null;

  constructor(message: string, statusCode: number | null = null, body: string | null = null) {
    super(message);
    this.statusCode = statusCode;
    this.body = body;
  }
}

/** The requested model id is not in the registry. */
export class ModelNotFoundError extends FreeAIPIError {}
