"""LLM role resolution -- credentials gate, factory inputs, observer choice.

Each role's section resolves into an ``LLMClientConfig`` (or raises
``LLMNotConfiguredError`` naming the section), which ``build_llm_provider``
turns into an ``OpenAIProvider``; with ``[observability] enabled`` the
provider reports to a ``TracingObserver``, otherwise to a ``NullObserver``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from everos.component.llm import (
    LLMClientConfig,
    LLMNotConfiguredError,
    NullObserver,
    OpenAIProvider,
    build_llm_provider,
    observer_for,
    resolve_decider_config,
    resolve_llm_config,
    resolve_multimodal_config,
)
from everos.component.llm._tracing_observer import TracingObserver
from everos.config import Settings
from everos.config.settings import (
    LLMSettings,
    MultimodalSettings,
    ObservabilitySettings,
)


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


def test_raises_when_api_key_missing() -> None:
    with pytest.raises(LLMNotConfiguredError, match=r"\[llm\]"):
        resolve_llm_config(_settings(api_key=None))


def test_raises_when_base_url_missing() -> None:
    with pytest.raises(LLMNotConfiguredError, match=r"\[llm\]"):
        resolve_llm_config(_settings(base_url=None))


def test_the_factory_builds_an_openai_provider() -> None:
    settings = _settings()

    provider = build_llm_provider(
        resolve_llm_config(settings), observer=observer_for(settings)
    )

    assert isinstance(provider, OpenAIProvider)


def test_llm_section_reaches_the_factory() -> None:
    config = resolve_llm_config(
        _settings(
            timeout_seconds=30.0,
            extra={"extra_body": {"x": 1}},
            response_format_mode="json_object",
            invalid_reply_retries=3,
        )
    )

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
    ("enabled", "observer_type"),
    [(True, TracingObserver), (False, NullObserver)],
)
def test_observer_follows_observability(enabled: bool, observer_type: type) -> None:
    assert isinstance(observer_for(_settings(observability=enabled)), observer_type)


def test_multimodal_section_reaches_the_factory() -> None:
    config = resolve_multimodal_config(_settings())

    assert (config.model, config.api_key, config.base_url) == (
        "google/gemini-3.8-flash",
        "sk-mm",
        "https://mm.test",
    )


def test_decider_mode_follows_the_section_that_supplies_the_model() -> None:
    """Format support is a property of the model, so it is inherited with it."""
    settings = _settings(response_format_mode="prompt_only")
    assert resolve_decider_config(settings).response_format_mode == "prompt_only"

    settings.decider.model = "other-model"
    assert resolve_decider_config(settings).response_format_mode == "json_schema"
