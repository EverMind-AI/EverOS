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
    resolve_profile_config,
)
from everos.component.llm._tracing_observer import TracingObserver
from everos.config import Settings
from everos.config.settings import (
    LLMSettings,
    MultimodalSettings,
    ObservabilitySettings,
    ProfileSettings,
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


def test_an_unnamed_profile_model_is_llm_field_for_field() -> None:
    settings = _settings(timeout_seconds=75.0, extra={"extra_body": {"x": 1}})
    settings.profile = ProfileSettings(timeout_seconds=300.0, extra={"ignored": True})

    assert resolve_profile_config(settings) == resolve_llm_config(settings)


def test_a_named_profile_model_inherits_only_the_endpoint() -> None:
    settings = _settings(
        response_format_mode="prompt_only", extra={"extra_body": {"x": 1}}
    )
    settings.profile = ProfileSettings(
        model="claude-haiku-5-5", extra={"extra_body": {"thinking": True}}
    )

    config = resolve_profile_config(settings)

    assert (config.model, config.api_key, config.base_url) == (
        "claude-haiku-5-5",
        "sk-test",
        "https://example.test",
    )
    assert config.timeout == 240.0
    assert config.extra == {"extra_body": {"thinking": True}}
    assert config.response_format_mode == "json_schema"


def test_a_named_profile_model_takes_its_own_endpoint() -> None:
    settings = _settings()
    settings.profile = ProfileSettings(
        model="m", api_key=SecretStr("sk-profile"), base_url="https://profile.test"
    )

    config = resolve_profile_config(settings)

    assert (config.api_key, config.base_url) == ("sk-profile", "https://profile.test")


def test_a_named_profile_model_without_any_endpoint_names_its_section() -> None:
    settings = _settings(api_key=None)
    settings.profile = ProfileSettings(model="m")

    with pytest.raises(LLMNotConfiguredError, match=r"\[profile\]"):
        resolve_profile_config(settings)
