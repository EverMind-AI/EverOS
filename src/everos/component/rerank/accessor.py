"""Process-wide rerank capability accessor.

Lazy singleton mirror of
:func:`everos.component.embedding.accessor.get_embedding_capability`: first
call reads settings, attempts to build a rerank provider, and wraps the
outcome (provider or ``None``) in a :class:`RerankCapability`. Subsequent
calls return the cached instance.

Rerank is a Tier-3 optional provider — call sites either go through
:func:`get_rerank_capability` and call ``.require()`` when rerank is
mandatory, or check ``.available`` and skip reranking when it is not.
"""

from __future__ import annotations

from everos.config import load_settings
from everos.core.observability.logging import get_logger

from .capability import RerankCapability
from .factory import build_rerank_provider

logger = get_logger(__name__)


_capability: RerankCapability | None = None


def get_rerank_capability() -> RerankCapability:
    """Return the process-wide :class:`RerankCapability`. Never raises.

    Lazy singleton: the first call builds and caches a capability from
    current settings — ``available`` is ``False`` when the provider cannot
    be built (e.g. missing model/base_url/api_key, unsupported provider
    name, malformed URL, …). Use this from search strategies, the health
    endpoint, the startup banner, and any other caller that needs to
    check "is rerank available?" without a hard dependency.

    Configuration failures (:class:`ValueError` from
    :func:`build_rerank_provider`) are logged at ``warning`` level: the
    downstream :class:`ProviderNotConfiguredError` message maps both
    "user hasn't configured it" and "user configured it wrong" onto the
    same HTTP 422, so the log line is the only place an operator can
    tell those two states apart.
    """
    global _capability
    if _capability is not None:
        return _capability
    try:
        provider = build_rerank_provider(load_settings().rerank)
    except ValueError as exc:
        logger.warning(
            "rerank_capability_build_failed",
            reason=str(exc),
        )
        provider = None
    _capability = RerankCapability(provider=provider)
    return _capability
