"""
LLM configuration management

Provides simple LLM configuration management
"""

from typing import Optional
from memory_layer.llm.openai_provider import OpenAIProvider
from memory_layer.llm.llm_provider import resolve_provider_env


def create_provider(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs,
) -> OpenAIProvider:
    """
    Create an OpenAI provider

    Args:
        model: Model name
        api_key: API key, if None use environment variable
        base_url: Base URL, if None use default value
        temperature: Temperature
        max_tokens: Maximum token count
        **kwargs: Additional parameters

    Returns:
        Configured OpenAIProvider instance
    """
    api_key, base_url = resolve_provider_env("openai", api_key=api_key, base_url=base_url)
    return OpenAIProvider(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def create_cheap_provider() -> OpenAIProvider:
    """Create a cheap provider (using gpt-4o-mini)"""
    return create_provider(model="gpt-4o-mini", temperature=0.3, max_tokens=1024)


def create_high_quality_provider() -> OpenAIProvider:
    """Create a high-quality provider (using gpt-4o)"""
    return create_provider(model="gpt-4o", temperature=0.7, max_tokens=4096)
