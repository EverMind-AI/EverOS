"""Embedding provider adapters (one provider per file).


Public surface:

- :class:`EmbeddingProvider` — Protocol every provider satisfies.
- :class:`EmbeddingServiceError` — provider-side failure.
- :class:`EmbeddingError` — backward-compat alias for ``EmbeddingServiceError``.
- :class:`OpenAIEmbeddingProvider` — concrete provider for any
  OpenAI-protocol embeddings endpoint (DeepInfra, vLLM, OpenAI, …).
- :class:`OllamaEmbeddingProvider` — local Ollama embedding provider
  with sensible defaults (``http://localhost:11434/v1``, dim=768).
- :func:`build_embedding_provider` — settings-driven factory.

External usage::

    from everos.component.embedding import build_embedding_provider
    provider = build_embedding_provider(settings.embedding)
    vec = await provider.embed("hello")
"""

from everos.core.errors import EmbeddingServiceError as EmbeddingServiceError

from .accessor import EmbeddingNotConfiguredError as EmbeddingNotConfiguredError
from .accessor import get_embedder as get_embedder
from .factory import build_embedding_provider as build_embedding_provider
from .ollama_provider import OllamaEmbeddingProvider as OllamaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider as OpenAIEmbeddingProvider
from .protocol import EmbeddingError as EmbeddingError
from .protocol import EmbeddingProvider as EmbeddingProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingNotConfiguredError",
    "EmbeddingProvider",
    "EmbeddingServiceError",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
    "get_embedder",
]
