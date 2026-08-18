"""Episode recaller — BM25 over ``episode_tokens`` + cosine ANN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import (
    Episode,
    Predicate,
    all_of,
    episode_repo,
    one_of,
)

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


logger = get_logger(__name__)
_FETCH_ALL_OWNER_LIMIT = 10_000


class EpisodeRecaller:
    """BM25 + vector recall over the LanceDB ``episode`` table."""

    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "episode"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: Predicate, *, limit: int
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
        rows = await episode_repo.sparse_search(
            terms,
            where,
            columns=Episode.BM25_FIELDS,
            limit=limit,
        )
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        rows = await episode_repo.dense_search(vector, where, limit=limit)
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]

    async def sparse_recall_as_child(
        self, query: str, where: Predicate, *, limit: int
    ) -> list[Candidate]:
        """Sparse recall returning episodes as MaxSim child candidates."""
        return _inject_parent_id(await self.sparse_recall(query, where, limit=limit))

    async def dense_recall_as_child(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]:
        """Dense recall (body vector ANN) returning as MaxSim children."""
        return _inject_parent_id(await self.dense_recall(vector, where, limit=limit))

    async def dense_recall_subject(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]:
        """ANN over the ``subject_vector`` column.

        Rows with ``subject_vector=None`` are naturally excluded by
        LanceDB ANN.
        """
        if not vector:
            return []
        rows = await episode_repo.dense_search(
            vector,
            where,
            limit=limit,
            vector_field="subject_vector",
        )
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]

    async def dense_recall_subject_as_child(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]:
        """Subject-vector ANN returning as MaxSim children."""
        candidates = await self.dense_recall_subject(vector, where, limit=limit)
        return _inject_parent_id(candidates)

    async def fetch_all_for_owner(self, where: Predicate) -> list[Candidate]:
        """Flat scan — all episodes for this owner, keyed by entry_id.

        Cluster membership matching in ``acluster_retrieve`` compares
        ``Candidate.id`` against ``Cluster.members``. Both are now
        episode entry_ids regardless of parent_type.

        The scan is capped at :data:`_FETCH_ALL_OWNER_LIMIT` to bound memory.
        Hitting the cap is logged because cluster matching may be incomplete.
        """
        rows = await episode_repo.search(where=where, limit=_FETCH_ALL_OWNER_LIMIT)
        if len(rows) == _FETCH_ALL_OWNER_LIMIT:
            logger.warning(
                "episode_owner_scan_truncated",
                limit=_FETCH_ALL_OWNER_LIMIT,
            )
        result: list[Candidate] = []
        for r in rows:
            entry_id = r.get("entry_id")
            if not isinstance(entry_id, str) or not entry_id:
                continue
            base = row_to_candidate(r, source="vector", score=0.0)
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
        self, entry_ids: list[str], where: Predicate
    ) -> list[Candidate]:
        """Fetch episodes by entry_id (for facts whose parent_id is an entry_id)."""
        if not entry_ids:
            return []
        full_where = all_of(where, one_of("entry_id", entry_ids))
        rows = await episode_repo.search(where=full_where, limit=len(entry_ids))
        return [row_to_candidate(r, source="vector", score=0.0) for r in rows]
