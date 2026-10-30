"""``build_llm_provider`` -- every config field reaches the provider."""

from __future__ import annotations

from everos.component.llm import (
    LLMClientConfig,
    NullObserver,
    OpenAIProvider,
    build_llm_provider,
)


def _config(**overrides: object) -> LLMClientConfig:
    fields: dict[str, object] = {
        "model": "m",
        "api_key": "k",
        "base_url": "https://x.invalid/v1",
    }
    return LLMClientConfig(**(fields | overrides))  # type: ignore[arg-type]  # test-only kwargs bag


def test_builds_openai_provider() -> None:
    assert isinstance(build_llm_provider(_config()), OpenAIProvider)


def test_passes_response_format_mode_and_retries() -> None:
    provider = build_llm_provider(
        _config(response_format_mode="prompt_only", invalid_reply_retries=4)
    )
    assert provider._response_format_mode == "prompt_only"
    assert provider._invalid_reply_retries == 4


def test_passes_endpoint_fields_to_the_sdk_client() -> None:
    provider = build_llm_provider(
        _config(timeout=12.5, max_retries=0, extra={"extra_body": {"x": 1}})
    )
    assert provider._client.timeout == 12.5
    assert provider._client.max_retries == 0
    assert str(provider._client.base_url) == "https://x.invalid/v1/"
    assert provider._extra == {"extra_body": {"x": 1}}


def test_passes_observer() -> None:
    observer = NullObserver()
    provider = build_llm_provider(_config(), observer=observer)
    assert provider._observer is observer


def test_defaults_to_null_observer() -> None:
    assert isinstance(build_llm_provider(_config())._observer, NullObserver)
