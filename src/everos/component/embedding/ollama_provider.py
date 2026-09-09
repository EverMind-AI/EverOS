"""Ollama embedding provider.

Ollama exposes an OpenAI-compatible ``/v1/embeddings`` endpoint at
``http://localhost:11434/v1``. This provider applies Ollama-specific
defaults so callers get a working configuration out of the box — no
need to remember the base URL or supply a dummy API key.

Usage::
    from everos.component.embedding.ollama_provider import OllamaEmbeddingProvider

    provider = OllamaEmbeddingProvider(
        model="nomic-embed-text",
        dim=768,
    )
    vec = await provider.embed("hello world")
"""

from __future__ import annotations

from .openai_provider import OpenAIEmbeddingProvider


_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_PLACEHOLDER_KEY = "ollama"


class OllamaEmbeddingProvider(OpenAIEmbeddingProvider):
    """Ollama embedding provider — OpenAI-compatible with local defaults.

    Same semantics as :class:`OpenAIEmbeddingProvider` except
    ``api_key`` and ``base_url`` fall back to Ollama's conventional
    values when omitted.

    Args:
        model: Ollama model id (e.g. ``"nomic-embed-text"``).
        api_key: API key; defaults to ``"ollama"``.
        base_url: Endpoint; defaults to ``http://localhost:11434/v1``.
        dim: Target vector dimension; defaults to 768 (Ollama's typical).
        timeout: Per-request timeout, seconds.
        max_retries: Retry budget.
        batch_size: Embeddings per ``/embeddings`` call.
        max_concurrent: Cap on in-flight chunked requests.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = _OLLAMA_PLACEHOLDER_KEY,
        base_url: str = _OLLAMA_DEFAULT_BASE_URL,
        dim: int = 768,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            dim=dim,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )
