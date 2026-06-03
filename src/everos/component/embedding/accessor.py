"""Process-wide embedding provider accessor.

Lazy singleton mirror of :func:`everos.component.llm.get_llm_client`:
first call reads settings and builds the OpenAI-protocol embedding
client; subsequent calls return the cached instance. Strategies and
other components that need a process-wide embedder import this rather
than threading the provider through their constructors.

Raises :class:`EmbeddingNotConfiguredError` when credentials are missing
so misconfiguration surfaces at the call site (or at app startup via a
lifespan provider) instead of silently degrading.
"""

from __future__ import annotations

from everos.config import load_settings
from everos.core.observability.logging import get_logger

from .factory import build_embedding_provider
from .protocol import EmbeddingProvider

logger = get_logger(__name__)


class EmbeddingNotConfiguredError(RuntimeError):
    """Raised when ``settings.embedding`` lacks ``model``/``api_key``/``base_url``."""


_embedder: EmbeddingProvider | None = None


def get_embedder() -> EmbeddingProvider:
    """Return the singleton :class:`EmbeddingProvider`.

    Raises:
        EmbeddingNotConfiguredError: When required settings fields are
            unset. See :func:`build_embedding_provider` for the exact
            keys.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        _embedder = build_embedding_provider(load_settings().embedding)
    except ValueError as exc:
        raise EmbeddingNotConfiguredError(str(exc)) from exc
    logger.info("embedder_built")
    return _embedder
