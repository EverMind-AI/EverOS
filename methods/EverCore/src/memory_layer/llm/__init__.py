"""
LLM providers module for memory layer.

This module provides LLM providers for the memory layer functionality.
"""

from memory_layer.llm.openai_provider import OpenAIProvider
from memory_layer.llm.protocol import LLMProvider
from memory_layer.llm.llm_provider import resolve_provider_env

__all__ = ["LLMProvider", "OpenAIProvider"]


def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Provider name (openai-compatible by default)
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM provider instance
    """
    provider_type = provider_type.lower()

    api_key, base_url = resolve_provider_env(
        provider_type,
        api_key=kwargs.pop("api_key", None),
        base_url=kwargs.pop("base_url", None),
    )

    return OpenAIProvider(
        provider_type=provider_type,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def create_provider_from_env(provider_type: str, **kwargs) -> LLMProvider:
    """
    Create LLM provider from environment variables.

    Args:
        provider_type: Provider name (openai-compatible by default)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLM provider instance
    """
    provider_type = provider_type.lower()

    api_key, base_url = resolve_provider_env(
        provider_type,
        api_key=kwargs.pop("api_key", None),
        base_url=kwargs.pop("base_url", None),
    )

    return OpenAIProvider(
        provider_type=provider_type,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
