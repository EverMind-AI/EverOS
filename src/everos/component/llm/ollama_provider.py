"""Ollama LLM provider for everos.

Ollama exposes an OpenAI-compatible API at ``http://localhost:11434/v1``
by default, with ``api-key`` accepting any non-empty value. This
provider extends :class:`OpenAIProvider` with Ollama-specific defaults
so callers don't need to remember the canonical base URL or dummy key.

Usage::
    settings = LLMSettings(
        model="llama3.1",
        base_url="http://localhost:11434/v1",  # or omit — Ollama default
        api_key=SecretStr("ollama"),            # or omit — Ollama default
    )
    provider = OllamaProvider(model=settings.model)
"""

from __future__ import annotations

from typing import Any

from .openai_provider import OpenAIProvider
from .protocol import ChatMessage, ChatResponse


_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_PLACEHOLDER_KEY = "ollama"


class OllamaProvider(OpenAIProvider):
    """Ollama LLM provider — OpenAI-compatible with local defaults.

    Identical to :class:`OpenAIProvider` except that ``api_key`` and
    ``base_url`` fall back to Ollama's conventional values when omitted.

    Args:
        model: Ollama model id (e.g. ``"llama3.1"``, ``"qwen2.5"``).
        api_key: API key; defaults to ``"ollama"`` (Ollama's placeholder).
        base_url: Endpoint; defaults to ``http://localhost:11434/v1``.
        timeout: Per-request timeout in seconds.
        temperature: Default sampling temperature.
        max_tokens: Default max-tokens cap.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = _OLLAMA_PLACEHOLDER_KEY,
        base_url: str | None = _OLLAMA_DEFAULT_BASE_URL,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
