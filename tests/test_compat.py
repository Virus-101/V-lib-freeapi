"""Generic OpenAI-compatible provider, catalogue parsing, and presets.

These tests matter most at the two seams: whether a vendor's catalogue parses
correctly, and whether the auto-enable rule (key present → provider appears)
fires. Both are where a config-driven design actually breaks.
"""

from __future__ import annotations

import pytest

from freeaipi.errors import ConfigurationError
from freeaipi.http import FakeResponse, MockTransport
from freeaipi.models import Capabilities, ModelInfo, Provider
from freeaipi.presets import PRESETS, ProviderSpec, enabled_specs, load_config_file, config_paths
from freeaipi.providers.openai_compat import (
    OpenAICompatProvider,
    _safe_json,
    fetch_catalogue,
    parse_catalogue,
)
from freeaipi.registry import create_default_registry


def spec(namespace="testvendor", key_env=None, **kw) -> ProviderSpec:
    base = dict(
        namespace=namespace,
        base_url="https://api.example.com/v1",
        key_env=key_env if key_env is not None else f"{namespace.upper()}_API_KEY",
        free=True,
    )
    base.update(kw)
    return ProviderSpec(**base)


@pytest.fixture(autouse=True)
def _test_keys(monkeypatch):
    """Provide credentials for every key_env the registry tests declare.

    Only namespaces that need one are listed; a test that wants the missing-key
    path passes ``key_env=""`` and asserts on the warning instead.
    """
    for name in ("TESTVENDOR", "ALPHA", "BETA", "GROQ", "V"):
        monkeypatch.setenv(f"{name}_API_KEY", f"{name.lower()}-key")


def make_provider(responses=None, **spec_kw):
    s = spec(**spec_kw)
    transport = MockTransport(responses or [])
    models = parse_catalogue([{"id": "test-model-70b", "context_length": 65536}], s)
    return transport, s, OpenAICompatProvider(
        models[0], transport, s, s.base_url, "test-key"
    )


class TestCatalogueParsing:
    def test_openai_shape(self):
        models = parse_catalogue(
            {"data": [{"id": "m1", "context_length": 32768, "max_completion_tokens": 4096}]},
            spec(),
        )
        assert [m.model_id for m in models] == ["m1"]
        assert models[0].capabilities.context_length == 32768

    def test_bare_list_shape(self):
        models = parse_catalogue([{"id": "m1"}], spec())
        assert len(models) == 1

    def test_skips_entries_without_id(self):
        models = parse_catalogue(
            [{"id": "good"}, {"context_length": 8000}, "not-a-dict", {"id": "also-good"}],
            spec(),
        )
        assert [m.model_id for m in models] == ["good", "also-good"]

    def test_odd_shapes_do_not_raise(self):
        for raw in (None, {}, {"data": None}, {"data": "nonsense"}, 42, "string"):
            assert parse_catalogue(raw, spec()) == []

    def test_namespace_sets_registry_key(self):
        models = parse_catalogue([{"id": "m1"}], spec(namespace="groq"))
        assert models[0].key == "groq/m1"
        assert models[0].label == "groq"

    def test_provider_kind_is_compat_while_label_is_vendor(self):
        m = parse_catalogue([{"id": "m1"}], spec(namespace="cerebras"))[0]
        assert m.provider is Provider.COMPAT
        assert m.label == "cerebras"

    def test_missing_context_defaults(self):
        m = parse_catalogue([{"id": "m1"}], spec())[0]
        assert m.capabilities.context_length == 32768

    def test_max_output_defaults_to_quarter_context(self):
        m = parse_catalogue([{"id": "m1", "context_length": 8000}], spec())[0]
        assert m.capabilities.max_output_tokens == 2000

    def test_vision_from_modalities(self):
        m = parse_catalogue(
            [{"id": "m1", "architecture": {"input_modalities": ["text", "image"]}}], spec()
        )[0]
        assert m.capabilities.vision is True

    def test_vision_from_name_fallback(self):
        m = parse_catalogue([{"id": "qwen2.5-vl-72b"}], spec())[0]
        assert m.capabilities.vision is True

    def test_tools_from_supported_parameters(self):
        m = parse_catalogue([{"id": "m1", "supported_parameters": ["tools"]}], spec())[0]
        assert m.capabilities.tools is True

    def test_tools_from_name_fallback(self):
        m = parse_catalogue([{"id": "qwen2.5-coder-32b"}], spec())[0]
        assert m.capabilities.tools is True

    def test_reasoning_from_name_fallback(self):
        m = parse_catalogue([{"id": "deepseek-r1"}], spec())[0]
        assert m.capabilities.reasoning is True

    def test_free_flag_carried_from_spec(self):
        assert parse_catalogue([{"id": "m"}], spec(free=True))[0].free is True
        assert parse_catalogue([{"id": "m"}], spec(free=False))[0].free is False

    def test_auth_required_always_true(self):
        """A generic endpoint is always registered behind an account."""
        assert parse_catalogue([{"id": "m"}], spec())[0].auth_required is True


class TestProviderRequests:
    def test_url_and_auth_header(self):
        transport, s, p = make_provider([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["url"] == "https://api.example.com/v1/chat/completions"
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer test-key"

    def test_string_message_coerced_not_iterated(self):
        """``list("hi")`` would send two one-character messages."""
        transport, s, p = make_provider([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["json"]["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_auth_header_when_spec_has_no_key(self):
        transport = MockTransport([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        s = spec(key_env="")
        models = parse_catalogue([{"id": "m1"}], s)
        p = OpenAICompatProvider(models[0], transport, s, s.base_url, "")
        p.chat("hi")
        assert "Authorization" not in transport.requests[0]["headers"]

    def test_missing_key_rejected_when_spec_declares_one(self):
        s = spec()
        transport = MockTransport()
        models = parse_catalogue([{"id": "m1"}], s)
        with pytest.raises(ConfigurationError) as ei:
            OpenAICompatProvider(models[0], transport, s, s.base_url, "")
        assert "TESTVENDOR_API_KEY" in str(ei.value)

    def test_tools_and_tool_choice_forwarded(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        transport, s, p = make_provider([
            {"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}]}
        ])
        p.chat("hi", tools=tools)
        body = transport.requests[0]["json"]
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"

    def test_json_mode_sets_response_format(self):
        transport, s, p = make_provider([
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi", json_mode=True)
        assert transport.requests[0]["json"]["response_format"] == {"type": "json_object"}

    def test_wire_model_id_used(self):
        transport, s, p = make_provider([
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        ])
        p.chat("hi")
        assert transport.requests[0]["json"]["model"] == "test-model-70b"


class TestProviderResponses:
    def test_content_usage_and_tier(self):
        transport, s, p = make_provider([{
            "choices": [{"message": {"content": "OK", "reasoning": "think"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }])
        r = p.chat("hi")
        assert r.content == "OK"
        assert r.reasoning == "think"
        assert r.usage.total_tokens == 5
        assert r.tier == "free"
        assert r.cost_usd == 0.0
        assert r.provider is Provider.COMPAT

    def test_tool_calls_parsed(self):
        transport, s, p = make_provider([{
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}}
            ]}, "finish_reason": "tool_calls"}]
        }])
        r = p.chat("go")
        assert r.tool_calls[0].arguments == {"a": 1}

    def test_error_envelope_raises(self):
        from freeaipi.errors import ProviderError

        transport, s, p = make_provider([{"error": {"message": "bad", "code": 400}}])
        with pytest.raises(ProviderError):
            p.chat("hi")

    def test_no_choices_raises(self):
        from freeaipi.errors import ProviderError

        transport, s, p = make_provider([{"unexpected": True}])
        with pytest.raises(ProviderError):
            p.chat("hi")


class TestProviderStreaming:
    def test_deltas_and_tool_fragments(self):
        frames = [
            'data: {"choices":[{"delta":{"content":"He"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        transport, s, p = make_provider([frames])
        chunks = list(p.stream("hi"))
        assert "".join(c.delta for c in chunks if c.delta) == "Hello"
        calls = [c.tool_call for c in chunks if c.tool_call]
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"a": 1}


class TestFetchCatalogue:
    def test_fetches_from_base_url(self):
        s = spec()
        transport = MockTransport([{"data": [{"id": "m1", "context_length": 8000}]}])
        models = fetch_catalogue(transport, s, "k")
        assert transport.requests[0]["url"] == "https://api.example.com/v1/models"
        assert [m.model_id for m in models] == ["m1"]

    def test_custom_models_url(self):
        s = spec(models_url="https://api.example.com/catalog")
        transport = MockTransport([{"data": [{"id": "m1"}]}])
        fetch_catalogue(transport, s, "k")
        assert transport.requests[0]["url"] == "https://api.example.com/catalog"

    def test_failure_returns_empty_not_raises(self):
        s = spec()
        transport = MockTransport([Exception("offline")])
        assert fetch_catalogue(transport, s, "k") == []

    def test_sends_auth_header_when_key_given(self):
        s = spec()
        transport = MockTransport([{"data": []}])
        fetch_catalogue(transport, s, "k")
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer k"


class TestPresetEnablement:
    def test_env_var_enables_provider(self):
        assert [s.namespace for s in enabled_specs({"GROQ_API_KEY": "x"})] == ["groq"]

    def test_no_env_vars_enables_none(self):
        assert enabled_specs({}) == []

    def test_whitespace_value_is_not_a_key(self):
        assert enabled_specs({"GROQ_API_KEY": "   "}) == []

    def test_multiple_keys_enable_multiple(self):
        specs = enabled_specs({"GROQ_API_KEY": "a", "MISTRAL_API_KEY": "b"})
        assert [s.namespace for s in specs] == ["groq", "mistral"]

    def test_every_preset_has_distinct_namespace(self):
        assert len({s.namespace for s in PRESETS.values()}) == len(PRESETS)

    def test_every_preset_has_distinct_key_env(self):
        envs = [s.key_env for s in PRESETS.values() if s.key_env]
        assert len(envs) == len(set(envs))

    def test_every_preset_has_base_url(self):
        assert all(s.base_url.startswith("http") for s in PRESETS.values())

    def test_paid_providers_are_marked_not_free(self):
        """A paid endpoint in the catalogue must not read as a free one."""
        for name in ("deepseek", "nous", "inception"):
            assert PRESETS[name].free is False, f"{name} should be free=False"

    def test_free_providers_are_marked_free(self):
        for name in ("groq", "cerebras", "sambanova", "novita"):
            assert PRESETS[name].free is True, f"{name} should be free=True"


class TestConfigFile:
    def test_loads_providers(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text('{"providers": [{"namespace": "v", "base_url": "http://x:1/v1", "free": true}]}')
        specs = load_config_file(f)
        assert specs[0].namespace == "v"
        assert specs[0].base_url == "http://x:1/v1"
        assert specs[0].free is True

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_config_file(tmp_path / "nope.json") == []

    def test_malformed_json_returns_empty(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not json")
        assert load_config_file(f) == []

    def test_entry_without_base_url_skipped(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text('{"providers": [{"namespace": "v"}, {"namespace": "w", "base_url": "http://w"}]}')
        specs = load_config_file(f)
        assert [s.namespace for s in specs] == ["w"]

    def test_explicit_models_listed(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text('{"providers": [{"namespace": "v", "base_url": "http://x", "models": ["a", "b"]}]}')
        assert load_config_file(f)[0].models == ("a", "b")

    def test_config_paths_from_env(self):
        assert config_paths({}) == []
        assert config_paths({"FREEAIPI_PROVIDERS": "/tmp/a.json"}) == [__import__("pathlib").Path("/tmp/a.json")]

    def test_config_paths_ignores_empty(self):
        assert config_paths({"FREEAIPI_PROVIDERS": "   "}) == []


class TestRegistryIntegration:
    def test_spec_models_used_when_catalogue_unreachable(self):
        """A private server with no /models endpoint works via explicit ids."""
        s = spec(namespace="private", key_env="", models=("my-70b",))
        transport = MockTransport([Exception("no catalogue endpoint")])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[s],
        )
        assert [m.key for m in reg.list()] == ["private/my-70b"]

    def test_models_filter_narrows_catalogue(self):
        s = spec(namespace="v", models=("m2",))
        transport = MockTransport([{"data": [{"id": "m1"}, {"id": "m2"}]}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[s],
        )
        assert [m.model_id for m in reg.list()] == ["m2"]

    def test_exclude_removes_ids(self):
        s = spec(namespace="v", exclude=("m1",))
        transport = MockTransport([{"data": [{"id": "m1"}, {"id": "m2"}]}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[s],
        )
        assert [m.model_id for m in reg.list()] == ["m2"]

    def test_namespace_filter_selects_vendor(self):
        transport = MockTransport([
            {"data": [{"id": "a"}]},
            {"data": [{"id": "b"}]},
        ])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False,
            compat_specs=[spec(namespace="alpha"), spec(namespace="beta")],
        )
        assert [m.key for m in reg.list(namespace="alpha")] == ["alpha/a"]
        assert [m.key for m in reg.list(namespace="beta")] == ["beta/b"]
        assert reg.list(namespace="gamma") == []

    def test_empty_catalogue_warns(self):
        s = spec(namespace="v")
        transport = MockTransport([{"data": []}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[s],
        )
        assert any("no models discovered" in w for w in reg.warnings)

    def test_unreachable_catalogue_warns_not_raises(self):
        """``fetch_catalogue`` swallows the failure, so the warning is the trace."""
        s = spec(namespace="v")
        transport = MockTransport([Exception("boom")])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[s],
        )
        assert reg.list() == []
        assert any("v" in w and "no models discovered" in w for w in reg.warnings)

    def test_missing_key_warns_and_other_providers_survive(self):
        """A spec with no key must not take the whole registry down with it."""
        transport = MockTransport([
            {"data": [{"id": "good-model"}]},
            {"data": [{"id": "also-good"}]},
        ])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False,
            compat_specs=[
                spec(namespace="keyless"),                       # no key set
                spec(namespace="alpha"),                         # key set by fixture
            ],
        )
        assert [m.key for m in reg.list()] == ["alpha/good-model"]
        assert any("keyless" in w for w in reg.warnings)

    def test_include_compat_false_skips(self):
        transport = MockTransport([{"data": [{"id": "a"}]}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, include_compat=False, compat_specs=[spec()],
        )
        assert reg.list() == []

    def test_vendors_listed_by_namespace(self):
        transport = MockTransport([
            {"data": [{"id": "a"}]},
            {"data": [{"id": "b"}]},
        ])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False,
            compat_specs=[spec(namespace="alpha"), spec(namespace="beta")],
        )
        assert reg.providers_in_use() == ["alpha", "beta"]

    def test_by_provider_groups_by_namespace(self):
        transport = MockTransport([{"data": [{"id": "a"}, {"id": "b"}]}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[spec(namespace="alpha")],
        )
        assert list(reg.by_provider()) == ["alpha"]

    def test_get_resolves_vendor_key(self):
        transport = MockTransport([{"data": [{"id": "m1"}]}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[spec(namespace="groq")],
        )
        info, _ = reg.get("groq/m1")
        assert info.model_id == "m1"

    def test_get_bare_id_when_unique(self):
        transport = MockTransport([{"data": [{"id": "m1"}]}])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False, compat_specs=[spec(namespace="groq")],
        )
        assert reg.get("m1")[0].label == "groq"


class TestCoercionHelpers:
    def test_safe_json_valid(self):
        assert _safe_json('{"a":1}') == {"a": 1}

    def test_safe_json_empty(self):
        assert _safe_json("") == {}

    def test_safe_json_malformed_preserves_raw(self):
        assert _safe_json("{broken") == {"_raw": "{broken"}

    def test_safe_json_non_dict(self):
        assert _safe_json("[1,2]") == {"_raw": "[1,2]"}


class TestAutoEnableEndToEnd:
    """The whole point of presets: set a key, the provider appears."""

    def test_key_in_env_registers_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        transport = MockTransport([
            {"data": [{"id": "llama-3.3-70b-versatile", "context_length": 131072}]},
        ])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False,
        )
        assert reg.providers_in_use() == ["groq"]
        assert reg.list()[0].key == "groq/llama-3.3-70b-versatile"

    def test_no_keys_registers_no_compat_providers(self, monkeypatch):
        for env in PRESETS:
            monkeypatch.delenv(PRESETS[env].key_env, raising=False)
        monkeypatch.delenv("FREEAIPI_PROVIDERS", raising=False)
        transport = MockTransport([])
        reg = create_default_registry(
            transport=transport, include_local=False, include_openrouter=False,
            include_pollinations=False,
        )
        assert reg.providers_in_use() == []
