"""AgentCase recaller — dual-column BM25 + cosine ANN.

The schema declares two BM25 columns (``task_intent_tokens`` —
retrieval anchor, primary — and ``approach_tokens`` — secondary
detail match). LanceDB's ``nearest_to_text`` searches one column at
a time, so we run the BM25 query twice in parallel and merge by row
id keeping the max score across the two columns. Vector recall is
single-shot.

Mirrors :class:`AgentSkillRecaller` structurally — both kinds share
the multi-BM25-column pattern.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.lancedb import AgentCase, get_table

from .base import (
    RecallerDeps,
    build_or_query_multi_column,
    cosine_score_from_distance,
    row_to_candidate,
)


class AgentCaseRecaller:
    """BM25 (dual-column) + vector recall over the LanceDB ``agent_case`` table."""

    kind: ClassVar[str] = "agent_case"
    everalgo_memory_type: ClassVar[str] = "case"
    text_field: ClassVar[str] = "task_intent"

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
            self._deps.tokenizer, query, AgentCase.BM25_FIELDS
        )
        if column_queries is None:
            return []
        table = await get_table(AgentCase.TABLE_NAME, AgentCase)

        async def _query_one(column: str) -> list[dict]:
            return (
                await table.query()
                .nearest_to_text(column_queries[column])
                .where(where)
                .limit(limit)
                .to_list()
            )

        per_column = await asyncio.gather(
            *(_query_one(col) for col in AgentCase.BM25_FIELDS),
        )
        # Merge by id, keep the max BM25 score across the two columns.
        # task_intent hits typically score higher (the retrieval anchor);
        # approach hits catch queries that match a step detail.
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
        table = await get_table(AgentCase.TABLE_NAME, AgentCase)
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
