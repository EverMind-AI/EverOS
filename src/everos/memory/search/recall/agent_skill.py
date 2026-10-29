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

from everos.memory.ports import AgentSkillIndex, Filter, all_of, any_of, contains

from .base import (
    RecallerDeps,
    cosine_score_from_distance,
    row_to_candidate,
)


class AgentSkillRecaller:
    """BM25 + vector recall over the agent-skill index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The agent-skill index port.
    """

    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"
    text_field: ClassVar[str] = "description"

    def __init__(self, deps: RecallerDeps, index: AgentSkillIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> list[Candidate]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Mirrors ``AgentCaseRecaller.sparse_recall`` — see there for
        rationale. One BooleanQuery per BM25 column; merge by id with
        max score.
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

    async def fetch_by_case_ids(
        self, case_ids: Sequence[str], where: Filter, *, limit: int
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
        hits = await self._index.find(full_where, limit=limit)
        return [row_to_candidate(h, source="vector", score=0.0) for h in hits]
