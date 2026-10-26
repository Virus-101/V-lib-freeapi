"""Transport layer: retries, rate-limit parsing, credential resolution.

Retry behaviour is time-sensitive, so the sleeps are patched to be instant. The
test asserts *that* a retry happened and *how many times*, not how long it took.
"""

from __future__ import annotations

import httpx
import pytest

from freeaipi.credentials import Credential, require_key, resolve_key
from freeaipi.errors import AuthenticationError, ConfigurationError, ProviderError, RateLimitError
from freeaipi.http import FakeResponse, MockTransport, Transport
from freeaipi.models import ModelInfo, Provider


class _FakeClient:
    """Stands in for ``httpx.Client`` in retry tests.

    Returns queued responses regardless of method/url, so a test can script a
    sequence (503, 503, 200) without a network server. An ``httpx.Response``
    doubles as its own context manager, so the streaming path works too.
    """

    def __init__(self, responses):
        self._responses = list(responses)

    def request(self, method, url, **kwargs):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def stream(self, method, url, **kwargs):
        return _StreamCtx(self.request(method, url, **kwargs))


class _StreamCtx:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make retry backoff instant so tests do not actually wait."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)


class TestRetryPolicy:
    def test_success_first_try_no_retry(self):
        calls = {"n": 0}
        responses = [httpx.Response(200, json={"ok": True})]

        client = _FakeClient(responses)
        original = client.request

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        client.request = counting
        t = Transport()
        t._client = client
        assert t.post_json("http://x", {}) == {"ok": True}
        assert calls["n"] == 1

    def test_retries_on_503_then_succeeds(self):
        t = Transport(max_retries=3)
        t._client = _FakeClient([
            httpx.Response(503, text="overloaded"),
            httpx.Response(503, text="overloaded"),
            httpx.Response(200, json={"ok": True}),
        ])
        assert t.post_json("http://x", {}) == {"ok": True}

    def test_gives_up_after_max_retries(self):
        """``max_retries=2`` means 1 initial attempt + 2 retries = 3 calls."""
        calls = {"n": 0}
        t = Transport(max_retries=2)
        client = _FakeClient([httpx.Response(500, text="boom") for _ in range(5)])
        original = client.request

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        client.request = counting
        t._client = client
        with pytest.raises(ProviderError) as ei:
            t.post_json("http://x", {})
        assert calls["n"] == 3
        assert "boom" in str(ei.value)

    def test_429_raises_rate_limit_not_provider_error(self):
        t = Transport(max_retries=1)
        t._client = _FakeClient([
            httpx.Response(429, text="slow down", headers={"Retry-After": "7"}),
            httpx.Response(429, text="slow down", headers={"Retry-After": "7"}),
        ])
        with pytest.raises(RateLimitError) as ei:
            t.post_json("http://x", {})
        assert ei.value.retry_after == 7.0

    def test_retry_after_header_used_as_delay(self):
        slept = []
        import freeaipi.http as http_mod
        original_sleep = http_mod.time.sleep
        http_mod.time.sleep = lambda s: slept.append(s)
        try:
            t = Transport(max_retries=2)
            t._client = _FakeClient([
                httpx.Response(429, text="quota", headers={"Retry-After": "3"}),
                httpx.Response(200, json={"ok": 1}),
            ])
            t.post_json("http://x", {})
        finally:
            http_mod.time.sleep = original_sleep
        assert slept == [3.0]

    def test_retry_after_parsed_from_body_text(self):
        """Pollinations puts the delay in the message, not the header."""
        slept = []
        import freeaipi.http as http_mod
        original_sleep = http_mod.time.sleep
        http_mod.time.sleep = lambda s: slept.append(s)
        try:
            t = Transport(max_retries=2)
            t._client = _FakeClient([
                httpx.Response(402, json={"status": 402, "error": {"message": "Free limit reached. Retry after 15s"}}),
                httpx.Response(200, json={"ok": 1}),
            ])
            t.post_json("http://x", {})
        finally:
            http_mod.time.sleep = original_sleep
        assert slept == [15.0]

    def test_exponential_backoff_when_no_hint(self):
        slept = []
        import freeaipi.http as http_mod
        original_sleep = http_mod.time.sleep
        http_mod.time.sleep = lambda s: slept.append(s)
        try:
            t = Transport(max_retries=3, retry_backoff=1.0)
            t._client = _FakeClient([
                httpx.Response(502, text="bad gateway"),
                httpx.Response(502, text="bad gateway"),
                httpx.Response(200, json={"ok": 1}),
            ])
            t.post_json("http://x", {})
        finally:
            http_mod.time.sleep = original_sleep
        assert slept == [1.0, 2.0]

    def test_auth_error_not_retried(self):
        calls = {"n": 0}
        t = Transport(max_retries=3)
        client = _FakeClient([httpx.Response(401, text="bad key") for _ in range(5)])
        original = client.request

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        client.request = counting
        t._client = client
        with pytest.raises(AuthenticationError):
            t.post_json("http://x", {})
        assert calls["n"] == 1


class TestErrorShapes:
    def test_404_carries_status_and_body(self):
        t = Transport()
        t._client = _FakeClient([httpx.Response(404, text="not found here")])
        with pytest.raises(ProviderError) as ei:
            t.get_json("http://x")
        assert ei.value.status_code == 404
        assert "not found here" in (ei.value.body or "")

    def test_non_json_success_raises_provider_error(self):
        t = Transport()
        t._client = _FakeClient([httpx.Response(200, text="<html>nope</html>")])
        with pytest.raises(ProviderError):
            t.get_json("http://x")

    def test_timeout_retries_then_raises(self):
        calls = {"n": 0}
        t = Transport(max_retries=2)
        client = _FakeClient([httpx.TimeoutException("timed out") for _ in range(5)])
        original = client.request

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        client.request = counting
        t._client = client
        with pytest.raises(ProviderError):
            t.get_json("http://x")
        assert calls["n"] == 3

    def test_connection_error_raises_provider_error(self):
        t = Transport()
        t._client = _FakeClient([httpx.ConnectError("refused")])
        with pytest.raises(ProviderError) as ei:
            t.get_json("http://x")
        assert "refused" in str(ei.value)


class TestStreamingErrors:
    def test_stream_401_raises_auth(self):
        t = Transport()
        t._client = _FakeClient([httpx.Response(401, text="unauthorized")])
        with pytest.raises(AuthenticationError):
            list(t.stream_lines("http://x", {}))

    def test_stream_429_raises_rate_limit(self):
        t = Transport(max_retries=0)
        t._client = _FakeClient([httpx.Response(429, text="quota")])
        with pytest.raises(RateLimitError):
            list(t.stream_lines("http://x", {}))

    def test_stream_retries_transient_then_yields(self):
        import httpx as _httpx

        t = Transport(max_retries=2)
        t._client = _FakeClient([
            _httpx.Response(503, text="down"),
            _httpx.Response(
                200,
                content=b"data: {}\n\ndata: [DONE]\n",
                request=_httpx.Request("POST", "http://x"),
            ),
        ])
        assert list(t.stream_lines("http://x", {})) == ["data: {}", "", "data: [DONE]"]


class TestCredentialResolution:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        c = resolve_key("arg-key", "openrouter")
        assert c.value == "arg-key"
        assert c.source == "explicit"

    def test_env_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        c = resolve_key(None, "openrouter")
        assert c.value == "env-key"
        assert c.source == "env:OPENROUTER_API_KEY"

    def test_absent_everywhere(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        c = resolve_key(None, "openrouter")
        assert c.present is False
        assert c.source == "none"

    def test_whitespace_env_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
        c = resolve_key(None, "openrouter")
        assert c.present is False

    def test_custom_env_var_name(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_VAR", "custom")
        c = resolve_key(None, "whatever", env_var="MY_CUSTOM_VAR")
        assert c.value == "custom"

    def test_key_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        f = tmp_path / "key.txt"
        f.write_text("file-key\n")
        c = resolve_key(None, "openrouter", key_file=f)
        assert c.value == "file-key"
        assert c.source.startswith("file:")

    def test_missing_key_file_is_absent(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        c = resolve_key(None, "openrouter", key_file=tmp_path / "nope.txt")
        assert c.present is False

    def test_require_key_raises_with_env_name(self):
        with pytest.raises(ConfigurationError) as ei:
            require_key(Credential("", "none"), "openrouter")
        assert "OPENROUTER_API_KEY" in str(ei.value)

    def test_require_key_returns_value(self):
        assert require_key(Credential("k", "env:X"), "openrouter") == "k"

    def test_str_never_contains_value(self):
        c = Credential("sk-or-v1-abcdef123456", "explicit")
        assert "abcdef" not in str(c)
        assert "abcdef" not in repr(c)

    def test_short_key_still_redacted(self):
        c = Credential("abc", "explicit")
        assert "abc" not in str(c)
        assert "len=3" in str(c)


class TestFakeResponse:
    def test_dict_body(self):
        r = FakeResponse(200, {"a": 1})
        assert r.json() == {"a": 1}
        assert r.status_code == 200

    def test_list_body(self):
        r = FakeResponse(200, [1, 2])
        assert r.json() == [1, 2]

    def test_text_body(self):
        r = FakeResponse(200, text='{"b": 2}')
        assert r.json() == {"b": 2}

    def test_headers_present(self):
        assert "content-type" in FakeResponse(200, {}).headers


class TestMockTransport:
    def test_queues_responses_in_order(self):
        m = MockTransport([{"a": 1}, {"b": 2}])
        assert m._as_response(m._next()).json() == {"a": 1}
        assert m._as_response(m._next()).json() == {"b": 2}

    def test_records_requests(self):
        m = MockTransport([{"ok": 1}])
        m.post_json("http://x", {"q": 1}, {"H": "v"})
        assert m.requests[0]["json"] == {"q": 1}
        assert m.requests[0]["headers"] == {"H": "v"}

    def test_raises_when_exhausted(self):
        m = MockTransport([])
        with pytest.raises(ProviderError):
            m._next()

    def test_exception_entry_raises(self):
        m = MockTransport([RuntimeError("boom")])
        with pytest.raises(RuntimeError):
            m._request("GET", "http://x")

    def test_invalid_entry_type_rejected(self):
        m = MockTransport([42])
        with pytest.raises(TypeError):
            m._request("GET", "http://x")
