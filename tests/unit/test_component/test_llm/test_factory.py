"""``build_llm_provider`` — settings validation + provider build."""

from __future__ import annotations

import importlib

import pytest
from pydantic import SecretStr

from everos.component.llm import build_llm_provider
from everos.component.llm.openai_provider import OpenAIProvider
from everos.config.settings import LLMSettings

_factory_mod = importlib.import_module("everos.component.llm.factory")


def test_raises_when_api_key_missing() -> None:
    s = LLMSettings(model="m", api_key=None, base_url="https://x")
    with pytest.raises(ValueError, match="EVEROS_LLM__API_KEY"):
        build_llm_provider(s)


def test_raises_when_base_url_missing() -> None:
    s = LLMSettings(model="m", api_key=SecretStr("k"), base_url=None)
    with pytest.raises(ValueError, match="EVEROS_LLM__BASE_URL"):
        build_llm_provider(s)


def test_builds_openai_provider() -> None:
    s = LLMSettings(model="m", api_key=SecretStr("k"), base_url="https://x")
    p = build_llm_provider(s)
    assert isinstance(p, OpenAIProvider)


def test_passes_configured_timeout_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_provider(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(_factory_mod, "OpenAIProvider", fake_provider)
    settings = LLMSettings(
        model="m",
        api_key=SecretStr("k"),
        base_url="https://x",
        timeout_seconds=135.0,
    )

    provider = _factory_mod.build_llm_provider(settings)

    assert provider is sentinel
    assert captured["timeout"] == 135.0
