"""Deterministic embedding provider double."""

from __future__ import annotations

__all__ = ["FakeEmbeddingProvider"]

import hashlib
from collections.abc import Sequence


class FakeEmbeddingProvider:
    """Embeds text into a stable unit-free vector derived from its hash.

    Equal texts embed equally and different texts almost surely differ;
    ``calls`` records every text embedded, in order.

    Args:
        dim: Vector dimension.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)
