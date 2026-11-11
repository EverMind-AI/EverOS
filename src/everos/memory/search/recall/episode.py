"""Episode recaller -- BM25 + cosine ANN over the episode index."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everos.memory.models import Episode
from everos.memory.ports import EpisodeIndex, Filter

from .base import (
    RecallerDeps,
    RecallResult,
    from_hits,
    stable_sparse_top,
)


class EpisodeRecaller:
    """BM25 + vector recall over the episode index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The episode index port.
    """

    kind: ClassVar[str] = "episode"

    def __init__(self, deps: RecallerDeps, index: EpisodeIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        """BM25 recall via OR-mode BooleanQuery.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token (typically the partition owner's own name on
        owner-scoped corpora) cannot poison the entire query.
        Mirrors enterprise's ``bool.should + minimum_should_match=1``
        ES design.

        The cut is :func:`~.base.stable_sparse_top`'s: score ties at ``limit``
        are settled by id, so a repeated search recalls the same list.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await stable_sparse_top(
            lambda n: self._index.sparse_search(terms, where, limit=n), limit
        )
        return from_hits(hits, source="keyword")

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return from_hits(hits, source="vector")

    async def has_any(self, where: Filter) -> bool:
        """Whether any episode matches ``where`` (one row is read at most)."""
        return bool(await self._index.find(where, limit=1))
