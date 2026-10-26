"""HTTP transport.

httpx is used over urllib for one reason: streaming. The SSE paths in the
providers read response lines as they arrive, and urllib's ``urlopen`` buffers
in a way that makes that awkward. Retry and rate-limit policy lives here so no
provider reimplements it.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

import httpx

from .errors import AuthenticationError, ProviderError, RateLimitError

#: Statuses worth retrying.
RETRYABLE_STATUS = {402, 408, 425, 429, 500, 502, 503, 504}

#: The subset that means "you hit a quota", as opposed to "the server is having
#: a bad moment". Only these become :class:`RateLimitError`; a 500 that survives
#: its retries is a :class:`ProviderError`, and calling it a rate limit sends the
#: caller off to sleep and retry something that will never succeed.
QUOTA_STATUS = {402, 429}


def _quota_or_provider(
    status: int, url: str, body: str, headers: httpx.Headers
) -> RateLimitError | ProviderError:
    """Build the right error for a status that exhausted its retries."""
    if status in QUOTA_STATUS:
        return RateLimitError(
            f"{status} from {url}: {body[:300]}",
            retry_after=Transport._retry_after_seconds(headers, body),
        )
    return ProviderError(
        f"{status} from {url}: {body[:300]}", status_code=status, body=body
    )


class Transport:
    """Thin httpx wrapper with retry, rate-limit parsing and SSE streaming."""

    def __init__(
        self,
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        respect_retry_after: bool = True,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.respect_retry_after = respect_retry_after
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _retry_after_seconds(headers: httpx.Headers, body: str) -> float | None:
        """Extract a retry hint from ``Retry-After`` or a JSON error body.

        Pollinations does not send ``Retry-After``; it puts
        ``"Retry after 15s"`` in the error message. Both are parsed because the
        difference is the difference between sleeping the right amount and
        guessing.
        """
        ra = headers.get("retry-after") or headers.get("x-ratelimit-reset")
        if ra:
            try:
                return max(0.0, float(ra))
            except ValueError:
                pass  # an HTTP-date, or something non-numeric
        if body:
            import re

            m = re.search(r"retry after (\d+)", body, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return None

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Perform a request, retrying on transient and quota failures.

        ``timeout`` overrides the client default for this call only. Local
        discovery passes ~2 s so a dead port is skipped quickly instead of
        waiting out the 300 s chat timeout.
        """
        client = self._client
        if timeout is not None:
            client = client.copy(timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)))
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = client.request(method, url, headers=headers, json=json_body, params=params)
            except httpx.TimeoutException as exc:
                if attempt > self.max_retries:
                    raise ProviderError(f"Request to {url} timed out after {attempt} attempts") from exc
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
                continue
            except httpx.HTTPError as exc:
                raise ProviderError(f"Network error calling {url}: {exc}") from exc

            if resp.status_code < 400:
                return resp

            if resp.status_code in (401, 403):
                raise AuthenticationError(
                    f"{resp.status_code} from {url}: {resp.text[:300]}"
                )

            if resp.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                delay = (
                    self._retry_after_seconds(resp.headers, resp.text)
                    if self.respect_retry_after
                    else None
                )
                if delay is None:
                    delay = self.retry_backoff * (2 ** (attempt - 1))
                time.sleep(delay)
                continue

            if resp.status_code in RETRYABLE_STATUS:
                raise _quota_or_provider(resp.status_code, url, resp.text, resp.headers)

            raise ProviderError(
                f"{resp.status_code} from {url}: {resp.text[:500]}",
                status_code=resp.status_code,
                body=resp.text,
            )

    def post_json(
        self,
        url: str,
        json_body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST JSON and return the decoded body."""
        resp = self._request("POST", url, headers=headers, json_body=json_body)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"Non-JSON response from {url}: {resp.text[:300]}", status_code=resp.status_code
            ) from exc

    def get_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """GET and return the decoded body.

        ``timeout`` overrides the client default for this call. Local discovery
        passes ~2 s so a dead port is skipped quickly rather than waiting out
        the 300 s chat timeout.
        """
        resp = self._request("GET", url, headers=headers, params=params, timeout=timeout)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"Non-JSON response from {url}: {resp.text[:300]}", status_code=resp.status_code
            ) from exc

    def stream_lines(
        self,
        url: str,
        json_body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Iterable[str]:
        """POST and yield response lines as they arrive.

        Lines are yielded raw (including the ``data:`` prefix) and the caller
        parses. Parsing here would mean every provider duplicates the SSE
        framing rules.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                with self._client.stream(
                    "POST", url, headers=headers, json=json_body
                ) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", errors="replace")
                        if resp.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                            delay = (
                                self._retry_after_seconds(resp.headers, body)
                                if self.respect_retry_after
                                else None
                            )
                            time.sleep(delay if delay is not None else self.retry_backoff)
                            continue
                        if resp.status_code in (401, 403):
                            raise AuthenticationError(
                                f"{resp.status_code} from {url}: {body[:300]}"
                            )
                        if resp.status_code in RETRYABLE_STATUS:
                            raise _quota_or_provider(resp.status_code, url, body, resp.headers)
                        raise ProviderError(
                            f"{resp.status_code} from {url}: {body[:500]}",
                            status_code=resp.status_code,
                            body=body,
                        )
                    for line in resp.iter_lines():
                        yield line
                    return
            except httpx.TimeoutException as exc:
                if attempt > self.max_retries:
                    raise ProviderError(f"Stream to {url} timed out") from exc
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

    def post_raw(
        self, url: str, json_body: dict[str, Any], headers: dict[str, str] | None = None
    ) -> tuple[int, str]:
        """POST and return ``(status_code, text)`` without raising on non-2xx.

        Used by providers whose error contract is not JSON (and by tests).
        """
        resp = self._request("POST", url, headers=headers, json_body=json_body)
        return resp.status_code, resp.text

    def json_dumps(self, obj: Any) -> str:
        return json.dumps(obj)


class FakeResponse:
    """Minimal stand-in for ``httpx.Response`` for tests.

    Exposes exactly the four attributes the transport reads: ``status_code``,
    ``text``, ``json()`` and ``headers``. A mock returns these instead of real
    responses, so provider code paths run unchanged.
    """

    def __init__(self, status_code: int = 200, body: Any = None, text: str | None = None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else ("" if body is None else _dumps(body))
        self.headers = {"content-type": "application/json"}

    def json(self) -> Any:
        if self._body is not None:
            return self._body
        return __import__("json").loads(self.text)


def _dumps(obj: Any) -> str:
    return __import__("json").dumps(obj)


class MockTransport(Transport):
    """A transport that answers from a queue instead of the network.

    Inject canned responses without monkeypatching httpx, so a provider's
    request-shaping code is exercised exactly as it runs in production. A queue
    entry may be:

    * a ``dict`` — becomes a ``200`` response with that JSON body
    * a ``FakeResponse`` — used verbatim (for non-200 status tests)
    * an ``Exception`` instance — raised, to test the error path
    * a ``str`` / ``list[str]`` — yielded as raw stream lines
    """

    def __init__(self, responses: list[Any] | None = None, **kwargs: Any):
        # Deliberately do not call super().__init__ — there is no client to open.
        self.responses = list(responses or [])
        self.requests: list[dict[str, Any]] = []
        self.timeout = kwargs.get("timeout", 300.0)
        self.max_retries = 0
        self.retry_backoff = 0.0
        self.respect_retry_after = False

    def close(self) -> None:  # pragma: no cover - nothing to close
        pass

    def _next(self) -> Any:
        if not self.responses:
            raise ProviderError("MockTransport ran out of canned responses")
        return self.responses.pop(0)

    def _as_response(self, item: Any) -> FakeResponse:
        if isinstance(item, FakeResponse):
            return item
        if isinstance(item, (dict, list)):
            return FakeResponse(200, item)
        raise TypeError(f"MockTransport cannot use {type(item).__name__} as a response")

    def _request(self, method, url, *, headers=None, json_body=None, params=None, timeout=None):
        self.requests.append(
            {"method": method, "url": url, "headers": headers or {}, "json": json_body or {}}
        )
        item = self._next()
        if isinstance(item, Exception):
            raise item
        return self._as_response(item)

    def stream_lines(self, url, json_body, headers=None):
        self.requests.append({"method": "POST", "url": url, "headers": headers or {}, "json": json_body or {}})
        item = self._next()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            yield item
        elif isinstance(item, (list, tuple)):
            yield from item
        else:
            # A dict here is a single SSE frame's JSON payload.
            yield f"data: {_dumps(item)}\n"
