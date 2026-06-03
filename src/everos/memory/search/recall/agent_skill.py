"""AgentSkill recaller — dual-column BM25 + cosine ANN.

The skill schema declares two BM25 columns
(``description_tokens`` + ``content_tokens``). LanceDB's
``nearest_to_text`` searches one column at a time, so we run the query
twice and merge by row id keeping the max score. Vector recall is
single-shot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.lancedb import AgentSkill, get_table

from .base import (
    RecallerDeps,
    build_or_query_multi_column,
    cosine_score_from_distance,
    row_to_candidate,
)


def _q(value: str) -> str:
    return value.replace("'", "''")


class AgentSkillRecaller:
    """BM25 + vector recall over the LanceDB ``agent_skill`` table."""

    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"
    text_field: ClassVar[str] = "description"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Mirrors ``AgentCaseRecaller.sparse_recall`` — see there for
        rationale. One BooleanQuery per BM25 column; merge by id with
        max score.
        """
        column_queries = build_or_query_multi_column(
            self._deps.tokenizer, query, AgentSkill.BM25_FIELDS
        )
        if column_queries is None:
            return []
        table = await get_table(AgentSkill.TABLE_NAME, AgentSkill)

        async def _query_one(column: str) -> list[dict]:
            return (
                await table.query()
                .nearest_to_text(column_queries[column])
                .where(where)
                .limit(limit)
                .to_list()
            )

        per_column = await asyncio.gather(
            *(_query_one(col) for col in AgentSkill.BM25_FIELDS),
        )
        # Merge by id, keep max BM25 score across the two columns.
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
        table = await get_table(AgentSkill.TABLE_NAME, AgentSkill)
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

    async def fetch_by_case_ids(
        self, case_ids: Sequence[str], where: str, *, limit: int
    ) -> list[Candidate]:
        """Skills whose ``source_case_ids`` intersect ``case_ids``.
        Filter is ``array_has`` OR-ed per id (same as
        ``filters._compile_op_clause`` for ``array_str``).

        ``score`` returns ``0.0`` — the manager re-attaches the max-pooled
        source-case score. ``source_case_ids`` rides in ``metadata`` so
        the manager can max-pool without a second fetch.
        """
        if not case_ids:
            return []
        table = await get_table(AgentSkill.TABLE_NAME, AgentSkill)
        clause = " OR ".join(f"array_has(source_case_ids, '{_q(c)}')" for c in case_ids)
        full_where = f"({where}) AND ({clause})"
        rows = await table.query().where(full_where).limit(limit).to_list()
        return [row_to_candidate(r, source="vector", score=0.0) for r in rows]
