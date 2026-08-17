"""Process-wide embedding capability accessor.

Lazy singleton for :class:`EmbeddingCapability`. The first call reads
settings and attempts to build an OpenAI-protocol embedding provider;
subsequent calls return the cached wrapper. Consumers that must have an
embedder call ``get_embedding_capability().require()``; consumers that
can degrade gracefully use ``.embed_or_none`` or check ``.available``.

There is deliberately no separate ``get_embedder()`` accessor: routing
every consumer through the capability keeps a single provider (and its
underlying ``AsyncOpenAI`` client + ``asyncio.Semaphore``) per process,
so the configured ``max_concurrent`` bound holds instead of silently
doubling.
"""

from __future__ import annotations

from everos.config import load_settings
from everos.core.observability.logging import get_logger

from .capability import EmbeddingCapability
from .factory import build_embedding_provider

logger = get_logger(__name__)


_capability: EmbeddingCapability | None = None


def get_embedding_capability() -> EmbeddingCapability:
    """Return the process-wide :class:`EmbeddingCapability`. Never raises.

    On the first call, builds and caches a capability from current
    settings — ``available`` is ``False`` when the provider cannot be
    built (missing fields, unsupported provider name, malformed URL, …).
    The build outcome is cached, so a later settings change requires a
    process restart to take effect.

    Configuration failures (:class:`ValueError` from
    :func:`build_embedding_provider`) are logged at ``warning`` level:
    the downstream :class:`ProviderNotConfiguredError` message maps both
    "user hasn't configured it" and "user configured it wrong" onto the
    same HTTP 422, so the log line is the only place an operator can
    tell those two states apart.
    """
    global _capability
    if _capability is not None:
        return _capability
    try:
        provider = build_embedding_provider(load_settings().embedding)
    except ValueError as exc:
        logger.warning(
            "embedding_capability_build_failed",
            reason=str(exc),
        )
        provider = None
    _capability = EmbeddingCapability(provider=provider)
    return _capability
