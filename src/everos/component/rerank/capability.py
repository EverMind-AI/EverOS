"""RerankCapability — soft-dependency wrapper for RerankProvider.

Parallel structure to :class:`everos.component.embedding.EmbeddingCapability`,
but without a soft-degrade accessor: rerank is a query-time enhancement,
not a write path. A caller either hard-requires rerank (raising
:class:`ProviderNotConfiguredError` -> HTTP 422 when missing) or chooses to
skip reranking entirely — there is no equivalent of ``embed_or_none``.
"""

from __future__ import annotations

from dataclasses import dataclass

from everos.core.errors import ProviderNotConfiguredError

from .protocol import RerankProvider


@dataclass(frozen=True)
class RerankCapability:
    """Wraps an optional RerankProvider with a hard-require API."""

    provider: RerankProvider | None

    @property
    def available(self) -> bool:
        """True iff a provider was successfully constructed."""
        return self.provider is not None

    def require(self) -> RerankProvider:
        """Return the provider, or raise :class:`ProviderNotConfiguredError`.

        Caller supplies its own ``feature``/``alternative_hint`` context by
        catching and re-raising if needed; this base call only identifies
        the missing provider as ``"rerank"``.
        """
        if self.provider is None:
            raise ProviderNotConfiguredError(provider="rerank")
        return self.provider
