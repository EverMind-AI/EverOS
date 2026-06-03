"""Factory for building an LLM provider from :class:`LLMSettings`."""

from __future__ import annotations

from everos.config import LLMSettings

from .openai_provider import OpenAIProvider
from .protocol import LLMClient


def build_llm_provider(settings: LLMSettings) -> LLMClient:
    """Build an OpenAI-compatible LLM provider from settings.

    Unwraps :class:`pydantic.SecretStr` here so downstream callers never
    touch the raw key directly. Fails fast if either ``api_key`` or
    ``base_url`` is missing — caller is expected to set them via
    ``.env`` / user toml / programmatic init before calling.

    Args:
        settings: The :class:`LLMSettings` slice from
            :func:`everos.config.load_settings`.

    Returns:
        A provider that structurally satisfies
        :class:`everalgo.llm.LLMClient` and can be passed to everalgo
        operators via ``llm=``.

    Raises:
        ValueError: If ``api_key`` or ``base_url`` is unset.
    """
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
