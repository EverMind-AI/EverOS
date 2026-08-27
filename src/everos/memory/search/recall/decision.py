"""Decision recaller — dual-column BM25 + cosine ANN.

The schema declares two BM25 columns (``decision_tokens`` — retrieval
anchor, primary — and ``reason_tokens`` — secondary why-match).
LanceDB's ``nearest_to_text`` searches one column at a time, so we
run the BM25 query twice in parallel and merge by row id keeping the
max score across the two columns. Vector recall is single-shot and
is fed only from the Decision body (cascade embeds that column).

Mirrors :class:`AgentCaseRecaller` structurally. HYBRID fusion for
this kind is :func:`everalgo.rank.fusion.rrf` in the manager —
``everalgo_memory_type`` is unused because Decision is not an
``arank`` ``memory_type``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.lancedb import Decision, get_table

from .base import (
    RecallerDeps,
    build_or_query_multi_column,
    cosine_score_from_distance,
    row_to_candidate,
)


class DecisionRecaller:
    """BM25 (dual-column) + vector recall over the LanceDB ``decision`` table."""

    kind: ClassVar[str] = "decision"
    everalgo_memory_type: ClassVar[str] = ""
    """Unused. Decision HYBRID fuses via ``rrf`` and never builds
    :class:`~everalgo.types.RankInput`."""
    text_field: ClassVar[str] = "decision"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token doesn't poison the column query (see
        ``EpisodeRecaller.sparse_recall``). One BooleanQuery is built
        per BM25 column (``MatchQuery`` is column-bound), then the
        two per-column result lists merge by id keeping the max score.
        """
        column_queries = build_or_query_multi_column(
            self._deps.tokenizer, query, Decision.BM25_FIELDS
        )
        if column_queries is None:
            return []
        table = await get_table(Decision.TABLE_NAME, Decision)

        async def _query_one(column: str) -> list[dict]:
            return (
                await table.query()
                .nearest_to_text(column_queries[column])
                .where(where)
                .limit(limit)
                .to_list()
            )

        per_column = await asyncio.gather(
            *(_query_one(col) for col in Decision.BM25_FIELDS),
        )
        # Merge by id, keep the max BM25 score across the two columns.
        # decision-body hits typically score higher (the retrieval
        # anchor); reason hits catch "why did we pick X" queries.
        best: dict[str, dict] = {}
        for rows in per_column:
            for r in rows:
                rid = r.get("id")
                if not isinstance(rid, str):
                    continue
                score = float(r.get("_score", 0.0))
                existing = best.get(rid)
                if existing is None or score > float(existing.get("_score", 0.0)):
                    merged = dict(r)
                    merged["_score"] = score
                    best[rid] = merged
        merged_rows = sorted(
            best.values(), key=lambda r: float(r.get("_score", 0.0)), reverse=True
        )[:limit]
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in merged_rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        table = await get_table(Decision.TABLE_NAME, Decision)
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
