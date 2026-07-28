"""EmbeddingCapability — soft-dependency wrapper for EmbeddingProvider.

The wrapper encapsulates ``Optional[EmbeddingProvider]`` so consumers
never see ``None`` directly. Two consumption modes:

- :meth:`embed_or_none` for soft-degrade write paths (cascade handlers
  that write ``vector=NULL`` when embedding is unavailable).
- :meth:`require` for hard-required paths (search VECTOR/HYBRID/AGENTIC)
  that raise :class:`ProviderNotConfiguredError` -> HTTP 422 when missing.

Design rationale: EverOS has multiple code paths that need to know
whether embedding is available. Rather than each callsite reading
settings and re-implementing the check, the check lives in
:func:`everos.component.embedding.accessor.get_embedding_capability`
(module-level singleton); every consumer asks the capability.
"""

from __future__ import annotations

from dataclasses import dataclass

from everos.core.errors import ProviderNotConfiguredError

from .protocol import EmbeddingProvider


@dataclass(frozen=True)
class EmbeddingCapability:
    """Wraps an optional EmbeddingProvider with soft / strict consumption modes."""

    provider: EmbeddingProvider | None

    @property
    def available(self) -> bool:
        """True iff a provider was successfully constructed."""
        return self.provider is not None

    async def embed_or_none(self, text: str) -> list[float] | None:
        """Embed ``text`` if available, else return ``None``.

        For write paths that can proceed without a vector (cascade
        handlers write ``vector=NULL`` on the LanceDB row).
        """
        if self.provider is None:
            return None
        return list(await self.provider.embed(text))

    def require(self) -> EmbeddingProvider:
        """Return the provider, or raise :class:`ProviderNotConfiguredError`.

        For paths that cannot proceed without embedding (search
        VECTOR/HYBRID/AGENTIC methods). Caller supplies its own
        ``feature``/``alternative_hint`` context by catching and
        re-raising if needed; this base call only identifies the
        missing provider as ``"embedding"``.
        """
        if self.provider is None:
            raise ProviderNotConfiguredError(provider="embedding")
        return self.provider
