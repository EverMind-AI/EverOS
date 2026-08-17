"""Embedding provider adapters (one provider per file).


Public surface:

- :class:`EmbeddingProvider` — Protocol every provider satisfies.
- :class:`EmbeddingServiceError` — provider-side failure.
- :class:`EmbeddingError` — backward-compat alias for ``EmbeddingServiceError``.
- :class:`EmbeddingCapability` — soft-dependency wrapper around an
  optional :class:`EmbeddingProvider` (``available`` / ``embed_or_none``
  / ``require``).
- :class:`OpenAIEmbeddingProvider` — concrete provider for any
  OpenAI-protocol embeddings endpoint (DeepInfra, vLLM, OpenAI, …).
- :func:`build_embedding_provider` — settings-driven factory.
- :func:`get_embedding_capability` — process-wide lazy singleton
  accessor for :class:`EmbeddingCapability`. There is no separate
  ``get_embedder`` accessor: consumers that need a provider call
  ``get_embedding_capability().require()``, which routes every caller
  through a single shared provider (and its single ``AsyncOpenAI``
  client + ``asyncio.Semaphore``).

External usage::

    from everos.component.embedding import build_embedding_provider
    provider = build_embedding_provider(settings.embedding)
    vec = await provider.embed("hello")
"""

from everos.core.errors import EmbeddingServiceError as EmbeddingServiceError

from .accessor import get_embedding_capability as get_embedding_capability
from .capability import EmbeddingCapability as EmbeddingCapability
from .factory import build_embedding_provider as build_embedding_provider
from .openai_provider import OpenAIEmbeddingProvider as OpenAIEmbeddingProvider
from .protocol import EmbeddingError as EmbeddingError
from .protocol import EmbeddingProvider as EmbeddingProvider

__all__ = [
    "EmbeddingCapability",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingServiceError",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
    "get_embedding_capability",
]
