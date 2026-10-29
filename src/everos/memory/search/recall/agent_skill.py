"""AgentSkill recaller — dual-column BM25 + cosine ANN.

The skill schema declares two BM25 columns
(``description_tokens`` + ``content_tokens``). LanceDB's
``nearest_to_text`` searches one column at a time, so we run the query
twice and merge by row id keeping the max score. Vector recall is
single-shot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from everos.memory.models import AgentSkill
from everos.memory.ports import (
    AgentSkillIndex,
    Filter,
    SearchHit,
    all_of,
    any_of,
    contains,
)

from .base import RecallerDeps, RecallResult, from_hits


class AgentSkillRecaller:
    """BM25 + vector recall over the agent-skill index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The agent-skill index port.
    """

    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"

    def __init__(self, deps: RecallerDeps, index: AgentSkillIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[AgentSkill]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Mirrors ``AgentCaseRecaller.sparse_recall`` — see there for
        rationale. One BooleanQuery per BM25 column; merge by id with
        max score.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return from_hits(hits, source="keyword")

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[AgentSkill]:
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return from_hits(hits, source="vector")

    async def bridged_from_cases(
        self, case_scores: Mapping[str, float], where: Filter, *, top_k: int
    ) -> RecallResult[AgentSkill]:
        """Skills learned from the given cases, scored by their best case.

        The case->skill bridge: each skill whose ``source_case_ids`` meets
        ``case_scores`` takes the highest score among those cases. Scores must
        be LLM-rerank relevance in ``[0, 1]`` to stay comparable with the
        direct dense pool -- never BM25 / fusion scores. One case can map to
        several skills, so the fetch allows a small fan-out per case.
        """
        if not case_scores:
            return RecallResult.empty()
        full_where = all_of(
            where,
            any_of(*(contains("source_case_ids", case_id) for case_id in case_scores)),
        )
        limit = max(top_k, len(case_scores) * 4)
        hits = await self._index.find(full_where, limit=limit)
        bridged = [
            SearchHit(
                stored=hit.stored,
                score=max(
                    (
                        case_scores[cid]
                        for cid in hit.stored.item.source_case_ids
                        if cid in case_scores
                    ),
                    default=0.0,
                ),
            )
            for hit in hits
        ]
        return from_hits(bridged, source="vector")
