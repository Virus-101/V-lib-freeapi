"""Exception hierarchy for freeaipi.

Every error the library raises is a subclass of :class:`FreeAIPIError`, so a
caller can catch the family with one ``except``. The subclasses exist because the
handling differs in kind: a rate limit wants a retry after a delay, an auth
error wants a configuration change, a provider error wants a fallback model.
"""


class FreeAIPIError(Exception):
    """Base class for every error raised by freeaipi."""


class ConfigurationError(FreeAIPIError):
    """A required credential or setting is missing or malformed.

    Raised before a request is made. Distinguished from :class:`ProviderError`
    because the fix is local (set an env var, start Ollama), not a retry.
    """


class AuthenticationError(FreeAIPIError):
    """The provider rejected the credential (HTTP 401/403)."""


class RateLimitError(FreeAIPIError):
    """The provider refused the request for quota reasons (HTTP 402/429).

    Carries the provider's own retry hint when it sent one, because the real
    free tiers publish exact windows (Pollinations: one reply per 30 s per IP).
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderError(FreeAIPIError):
    """The provider returned a non-success response that is not auth/quota.

    Carries the HTTP status and the provider's body so the caller can log or
    surface it. The body is kept as-is; it is never assumed to be JSON.
    """

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ModelNotFoundError(FreeAIPIError):
    """The requested model id is not in the registry."""
