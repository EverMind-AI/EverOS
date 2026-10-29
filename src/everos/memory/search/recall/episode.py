"""Episode recaller — BM25 over ``episode_tokens`` + cosine ANN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.memory.ports import EpisodeIndex, Filter, all_of, one_of

from .base import (
    RecallerDeps,
    cosine_score_from_distance,
    row_to_candidate,
)


def _inject_parent_id(candidates: list[Candidate]) -> list[Candidate]:
    """Wrap candidates with ``parent_id`` for MaxSim group-by-parent."""
    return [
        Candidate(
            id=c.id,
            score=c.score,
            source=c.source,
            metadata={**c.metadata, "parent_id": c.metadata.get("entry_id", c.id)},
        )
        for c in candidates
    ]


class EpisodeRecaller:
    """BM25 + vector recall over the episode index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The episode index port.
    """

    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "episode"

    def __init__(self, deps: RecallerDeps, index: EpisodeIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> list[Candidate]:
        """BM25 recall via OR-mode BooleanQuery.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token (typically the partition owner's own name on
        owner-scoped corpora) cannot poison the entire query.
        Mirrors enterprise's ``bool.should + minimum_should_match=1``
        ES design.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return []
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return [
            row_to_candidate(h, source="keyword", score=h.score or 0.0) for h in hits
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        hits = await self._index.dense_search(vector, where, limit=limit)
        return [
            row_to_candidate(
                h,
                source="vector",
                score=cosine_score_from_distance(h.fields.get("_distance")),
            )
            for h in hits
        ]

    async def sparse_recall_as_child(
        self, query: str, where: Filter, *, limit: int
    ) -> list[Candidate]:
        """Sparse recall returning episodes as MaxSim child candidates."""
        return _inject_parent_id(await self.sparse_recall(query, where, limit=limit))

    async def dense_recall_as_child(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Candidate]:
        """Dense recall (body vector ANN) returning as MaxSim children."""
        return _inject_parent_id(await self.dense_recall(vector, where, limit=limit))

    async def dense_recall_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Candidate]:
        """ANN over the ``subject_vector`` column.

        Rows with ``subject_vector=None`` are naturally excluded by
        LanceDB ANN.
        """
        if not vector:
            return []
        hits = await self._index.dense_search_subject(vector, where, limit=limit)
        return [
            row_to_candidate(
                h,
                source="vector",
                score=cosine_score_from_distance(h.fields.get("_distance")),
            )
            for h in hits
        ]

    async def dense_recall_subject_as_child(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Candidate]:
        """Subject-vector ANN returning as MaxSim children."""
        candidates = await self.dense_recall_subject(vector, where, limit=limit)
        return _inject_parent_id(candidates)

    async def fetch_all_for_owner(self, where: Filter) -> list[Candidate]:
        """Flat scan — all episodes for this owner, keyed by entry_id.

        Cluster membership matching in ``acluster_retrieve`` compares
        ``Candidate.id`` against ``Cluster.members``. Both are now
        episode entry_ids regardless of parent_type.

        No ``limit`` — the full owner partition is required for cluster
        membership matching.
        """
        result: list[Candidate] = []
        for hit in await self._index.find(where):
            entry_id = hit.fields.get("entry_id")
            if not isinstance(entry_id, str) or not entry_id:
                continue
            base = row_to_candidate(hit, source="vector", score=0.0)
            result.append(
                Candidate(
                    id=entry_id,
                    score=0.0,
                    source="vector",
                    metadata={**base.metadata, "episode_id": base.id},
                )
            )
        return result

    async def fetch_by_entry_ids(
        self, entry_ids: list[str], where: Filter
    ) -> list[Candidate]:
        """Fetch episodes by entry_id (for facts whose parent_id is an entry_id)."""
        if not entry_ids:
            return []
        full_where = all_of(where, one_of("entry_id", entry_ids))
        hits = await self._index.find(full_where, limit=len(entry_ids))
        return [row_to_candidate(h, source="vector", score=0.0) for h in hits]
