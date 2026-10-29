"""LLM client accessors -- credentials gate, caching, factory and observer wiring.

Every accessor resolves its section into an ``LLMClientConfig`` and builds
the provider through ``build_llm_provider``; with ``[observability] enabled``
the provider reports to a ``TracingObserver``, otherwise to a
``NullObserver``.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import SecretStr

from everos.component.llm import (
    LLMClientConfig,
    LLMNotConfiguredError,
    NullObserver,
    OpenAIProvider,
)
from everos.component.llm._tracing_observer import TracingObserver
from everos.config import Settings
from everos.config.settings import (
    LLMSettings,
    MultimodalSettings,
    ObservabilitySettings,
)

_client_mod = importlib.import_module("everos.component.llm.client")


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_llm_client", "_decider_client", "_multimodal_client"):
        monkeypatch.setattr(_client_mod, name, None)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Stub the ``load_settings`` reference bound inside the client module."""
    monkeypatch.setattr(_client_mod, "load_settings", lambda: settings)


def _settings(
    *,
    api_key: str | None = "sk-test",
    base_url: str | None = "https://example.test",
    observability: bool = False,
    **llm_fields: Any,
) -> Settings:
    return Settings(
        llm=LLMSettings(
            model="gpt-4.1-mini",
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=base_url,
            **llm_fields,
        ),
        multimodal=MultimodalSettings(
            api_key=SecretStr("sk-mm"), base_url="https://mm.test"
        ),
        observability=ObservabilitySettings(enabled=observability),
    )


def _capture_builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any]]:
    builds: list[tuple[Any, Any]] = []
    real = _client_mod.build_llm_provider

    def spy(config: LLMClientConfig, *, observer: Any = None) -> OpenAIProvider:
        builds.append((config, observer))
        return real(config, observer=observer)

    monkeypatch.setattr(_client_mod, "build_llm_provider", spy)
    return builds


def test_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, _settings(api_key=None))
    with pytest.raises(LLMNotConfiguredError, match=r"\[llm\]"):
        _client_mod.get_llm_client()


def test_raises_when_base_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, _settings(base_url=None))
    with pytest.raises(LLMNotConfiguredError, match=r"\[llm\]"):
        _client_mod.get_llm_client()


def test_returns_cached_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, _settings())
    builds = _capture_builds(monkeypatch)

    first = _client_mod.get_llm_client()
    second = _client_mod.get_llm_client()

    assert first is second
    assert isinstance(first, OpenAIProvider)
    assert len(builds) == 1


def test_llm_section_reaches_the_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(
        monkeypatch,
        _settings(
            timeout_seconds=30.0,
            extra={"extra_body": {"x": 1}},
            response_format_mode="json_object",
            invalid_reply_retries=3,
        ),
    )
    builds = _capture_builds(monkeypatch)

    _client_mod.get_llm_client()

    ((config, _),) = builds
    assert config == LLMClientConfig(
        model="gpt-4.1-mini",
        api_key="sk-test",
        base_url="https://example.test",
        timeout=30.0,
        extra={"extra_body": {"x": 1}},
        response_format_mode="json_object",
        invalid_reply_retries=3,
    )


@pytest.mark.parametrize(
    "accessor",
    ["get_llm_client", "get_decider_llm_client", "get_multimodal_llm_client"],
)
@pytest.mark.parametrize(
    ("enabled", "observer_type"),
    [(True, TracingObserver), (False, NullObserver)],
)
def test_observer_follows_observability(
    monkeypatch: pytest.MonkeyPatch,
    accessor: str,
    enabled: bool,
    observer_type: type,
) -> None:
    _patch_settings(monkeypatch, _settings(observability=enabled))
    builds = _capture_builds(monkeypatch)

    getattr(_client_mod, accessor)()

    ((_, observer),) = builds
    assert isinstance(observer, observer_type)


def test_multimodal_section_reaches_the_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, _settings())
    builds = _capture_builds(monkeypatch)

    client = _client_mod.get_multimodal_llm_client()

    ((config, _),) = builds
    assert isinstance(client, OpenAIProvider)
    assert (config.model, config.api_key, config.base_url) == (
        "google/gemini-3.8-flash",
        "sk-mm",
        "https://mm.test",
    )


def test_decider_mode_follows_the_section_that_supplies_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Format support is a property of the model, so it is inherited with it."""
    settings = _settings(response_format_mode="prompt_only")
    assert _client_mod.resolve_decider_config(settings).response_format_mode == (
        "prompt_only"
    )

    settings.decider.model = "other-model"
    assert _client_mod.resolve_decider_config(settings).response_format_mode == (
        "json_schema"
    )
