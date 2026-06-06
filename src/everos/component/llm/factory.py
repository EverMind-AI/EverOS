"""Factory for building an LLM provider from :class:`LLMSettings`.

Dispatches on ``settings.provider``:

- ``"openai"`` — :class:`OpenAIProvider` (requires ``base_url``).
- ``"litellm"`` — :class:`LiteLLMProvider` (routes via model-id
  prefix; ``base_url`` optional).
"""

from __future__ import annotations

from everos.config import LLMSettings

from .openai_provider import OpenAIProvider
from .protocol import LLMClient


def build_llm_provider(settings: LLMSettings) -> LLMClient:
    """Build an LLM provider from settings.

    Unwraps :class:`pydantic.SecretStr` here so downstream callers never
    touch the raw key directly.

    Args:
        settings: The :class:`LLMSettings` slice from
            :func:`everos.config.load_settings`.

    Returns:
        A provider that structurally satisfies
        :class:`everalgo.llm.LLMClient` and can be passed to everalgo
        operators via ``llm=``.

    Raises:
        ValueError: If required credentials are unset for the chosen
            provider.
    """
    if settings.provider == "litellm":
        return _build_litellm(settings)
    return _build_openai(settings)


def _build_openai(settings: LLMSettings) -> LLMClient:
    if settings.api_key is None:
        raise ValueError(
            "LLM api_key is not configured "
            "(set EVEROS_LLM__API_KEY or [llm] api_key in user toml)"
        )
    if not settings.base_url:
        raise ValueError(
            "LLM base_url is not configured "
            "(set EVEROS_LLM__BASE_URL or [llm] base_url in user toml)"
        )
    return OpenAIProvider(
        model=settings.model,
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
    )


def _build_litellm(settings: LLMSettings) -> LLMClient:
    from .litellm_provider import LiteLLMProvider

    return LiteLLMProvider(
        model=settings.model,
        api_key=(
            settings.api_key.get_secret_value()
            if settings.api_key is not None
            else None
        ),
        base_url=settings.base_url,
    )
