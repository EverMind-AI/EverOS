"""Embedding provider protocol.


The cascade worker / retrieval pipeline depend on a single small
contract: turn a string (or list of strings) into a fixed-dimension
vector. Whether the backend is OpenAI, vLLM, DeepInfra, Ollama, or a
local model is the provider's business — the contract is invariant.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from everos.core.errors import EmbeddingServiceError as EmbeddingServiceError

# Backward compat — old name still importable from this module.
EmbeddingError = EmbeddingServiceError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async embedding provider contract.

    ``dim`` is the post-truncation vector dimension every embed call
    returns. Providers that don't natively support dimension truncation
    must truncate client-side so callers see the declared shape.
    """

    dim: int

    async def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a ``[dim]`` vector."""
        ...

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings preserving input order.

        Implementations chunk by ``batch_size`` and bound in-flight
        requests by ``max_concurrent`` (both from settings). On failure,
        raises :class:`EmbeddingServiceError` — the worker treats it as a
        retryable / unrecoverable case per HTTP-status mapping.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed one interactive query.

        Semantically identical to :meth:`embed`; what differs is the
        concurrency budget it draws on. A search cannot answer until this
        returns, while a cascade embed is background work nobody waits on,
        so the two must not queue behind one another — with a single shared
        budget a burst of writes puts the user's query behind every one of
        them (issue #473: measured 48 s on a vector search while three
        writers were active, against 0.49 s for the keyword leg that needs
        no embedding at all).

        The default implementation falls back to :meth:`embed`, so a
        provider that does not separate the two still works.
        """
        return await self.embed(text)
