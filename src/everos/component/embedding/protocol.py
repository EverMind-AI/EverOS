"""Embedding provider protocol.


The cascade worker / retrieval pipeline depend on a single small
contract: turn a string (or list of strings) into a fixed-dimension
vector. Whether the backend is OpenAI, vLLM, DeepInfra, Ollama, or a
local model is the provider's business — the contract is invariant.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


class EmbeddingError(Exception):
    """Raised on any provider-side embedding failure.

    Wraps the upstream SDK exception via ``__cause__`` (PEP 3134) so
    diagnostic loggers preserve the original error chain.
    """


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
        raises :class:`EmbeddingError` — the worker treats it as a
        retryable / unrecoverable case per HTTP-status mapping.
        """
        ...
