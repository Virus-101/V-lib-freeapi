"""Shared fixtures.

Every test uses MockTransport, so the suite runs with the network disabled. A
fake model/provider pair is built through the real classes, meaning the
request-shaping code under test is the code that ships.
"""

from __future__ import annotations

import pytest

from freeaipi.http import FakeResponse, MockTransport
from freeaipi.models import Capabilities, ModelInfo, Provider
from freeaipi.providers.local import LocalProvider
from freeaipi.providers.openrouter import OpenRouterProvider
from freeaipi.providers.pollinations import PollinationsProvider


def make_model(
    provider=Provider.POLLINATIONS,
    model_id="openai-fast",
    provider_model_id="gpt-oss-20b",
    caps=None,
    auth_required=False,
) -> ModelInfo:
    return ModelInfo(
        model_id=model_id,
        provider=provider,
        provider_model_id=provider_model_id,
        name=f"{model_id} ({provider.value})",
        capabilities=caps or Capabilities(text=True, tools=True, reasoning=True),
        auth_required=auth_required,
    )


def make_pollinations(responses=None) -> tuple[MockTransport, PollinationsProvider]:
    transport = MockTransport(responses or [])
    model = make_model()
    return transport, PollinationsProvider(model, transport)


def make_openrouter(responses=None, key="sk-or-v1-test") -> tuple[MockTransport, OpenRouterProvider]:
    transport = MockTransport(responses or [])
    model = make_model(
        provider=Provider.OPENROUTER,
        model_id="nvidia/nemotron-3:free",
        provider_model_id="nvidia/nemotron-3:free",
    )
    return transport, OpenRouterProvider(model, transport, api_key=key)


def make_local(responses=None, kind="openai") -> tuple[MockTransport, LocalProvider]:
    transport = MockTransport(responses or [])
    model = make_model(
        provider=Provider.LOCAL,
        model_id="qwen2.5",
        provider_model_id="qwen2.5:7b",
    )
    return transport, LocalProvider(model, transport, "http://127.0.0.1:8080/v1", kind=kind)


@pytest.fixture
def transport():
    return MockTransport()


@pytest.fixture
def pollinations():
    return make_pollinations


@pytest.fixture
def openrouter():
    return make_openrouter


@pytest.fixture
def local():
    return make_local


@pytest.fixture
def fake_response():
    return FakeResponse
