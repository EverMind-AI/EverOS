"""AgentSkill recaller — dual-column BM25 + cosine ANN.

The skill schema declares two BM25 columns
(``description_tokens`` + ``content_tokens``). LanceDB's
``nearest_to_text`` searches one column at a time, so we run the query
twice and merge by row id keeping the max score. Vector recall is
single-shot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.index import (
    AgentSkill,
    Predicate,
    agent_skill_repo,
    all_of,
    any_of,
    contains,
)

from .base import (
    RecallerDeps,
    cosine_score_from_distance,
    row_to_candidate,
)


class AgentSkillRecaller:
    """BM25 + vector recall over the LanceDB ``agent_skill`` table."""

    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"
    text_field: ClassVar[str] = "description"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: Predicate, *, limit: int
    ) -> list[Candidate]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Mirrors ``AgentCaseRecaller.sparse_recall`` — see there for
        rationale. One BooleanQuery per BM25 column; merge by id with
        max score.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return []
        merged_rows = await agent_skill_repo.sparse_search(
            terms, where, columns=AgentSkill.BM25_FIELDS, limit=limit
        )
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in merged_rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        rows = await agent_skill_repo.dense_search(vector, where, limit=limit)
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]

    async def fetch_by_case_ids(
        self, case_ids: Sequence[str], where: Predicate, *, limit: int
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
        full_where = all_of(
            where,
            any_of(*(contains("source_case_ids", case_id) for case_id in case_ids)),
        )
        rows = await agent_skill_repo.search(where=full_where, limit=limit)
        return [row_to_candidate(r, source="vector", score=0.0) for r in rows]
