"""Embedding provider adapters (one provider per file).


Public surface:

- :class:`EmbeddingProvider` — Protocol every provider satisfies.
- :class:`EmbeddingServiceError` — provider-side failure (retryable).
- :class:`EmbeddingInputError` — the provider rejected the input (permanent).
- :class:`EmbeddingError` — backward-compat alias for ``EmbeddingServiceError``.
- :class:`OpenAIEmbeddingProvider` — concrete provider for any
  OpenAI-protocol embeddings endpoint (DeepInfra, vLLM, OpenAI, …).
- :func:`build_embedding_provider` — settings-driven factory.

There is no process-wide accessor: the application bootstrap builds one
provider (one ``AsyncOpenAI`` client + ``asyncio.Semaphore``, so the
configured ``max_concurrent`` bound holds) and injects it, or ``None``
when embedding is not configured.

External usage::

    from everos.component.embedding import build_embedding_provider
    provider = build_embedding_provider(settings.embedding)
    vec = await provider.embed("hello")
"""

from everos.core.errors import EmbeddingInputError as EmbeddingInputError
from everos.core.errors import EmbeddingServiceError as EmbeddingServiceError

from .factory import build_embedding_provider as build_embedding_provider
from .openai_provider import OpenAIEmbeddingProvider as OpenAIEmbeddingProvider
from .protocol import EmbeddingError as EmbeddingError
from .protocol import EmbeddingProvider as EmbeddingProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingProvider",
    "EmbeddingServiceError",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
]
