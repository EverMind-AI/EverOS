"""Episode recaller — BM25 over ``episode_tokens`` + cosine ANN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.lancedb import Episode, get_table

from .base import (
    RecallerDeps,
    build_or_query,
    cosine_score_from_distance,
    row_to_candidate,
)


def _q(value: str) -> str:
    return value.replace("'", "''")


class EpisodeRecaller:
    """BM25 + vector recall over the LanceDB ``episode`` table."""

    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "episode"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        """BM25 recall via OR-mode BooleanQuery.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token (typically the partition owner's own name on
        owner-scoped corpora) cannot poison the entire query.
        Mirrors enterprise's ``bool.should + minimum_should_match=1``
        ES design.
        """
        bq = build_or_query(self._deps.tokenizer, query, column=Episode.BM25_FIELDS[0])
        if bq is None:
            return []
        table = await get_table(Episode.TABLE_NAME, Episode)
        rows = (
            await table.query().nearest_to_text(bq).where(where).limit(limit).to_list()
        )
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        table = await get_table(Episode.TABLE_NAME, Episode)
        rows = (
            await table.query()
            .nearest_to(list(vector))
            .distance_type("cosine")
            .where(where)
            .limit(limit)
            .to_list()
        )
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]

    async def fetch_all_for_owner(self, where: str) -> list[Candidate]:
        """Flat scan — all episodes for this owner, keyed by entry_id.

        Cluster membership matching in ``acluster_retrieve`` compares
        ``Candidate.id`` against ``Cluster.members``. Both are now
        episode entry_ids regardless of parent_type.

        No ``limit`` — the full owner partition is required for cluster
        membership matching.
        """
        table = await get_table(Episode.TABLE_NAME, Episode)
        rows = await table.query().where(where).to_list()
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
        self, entry_ids: list[str], where: str
    ) -> list[Candidate]:
        """Fetch episodes by entry_id (for facts whose parent_id is an entry_id)."""
        if not entry_ids:
            return []
        table = await get_table(Episode.TABLE_NAME, Episode)
        quoted = ", ".join(f"'{_q(eid)}'" for eid in entry_ids)
        full_where = f"({where}) AND (entry_id IN ({quoted}))"
        rows = await table.query().where(full_where).limit(len(entry_ids)).to_list()
        return [row_to_candidate(r, source="vector", score=0.0) for r in rows]
