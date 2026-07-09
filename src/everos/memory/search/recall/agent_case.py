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

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.index import AgentCase, agent_case_repo

from .base import (
    RecallerDeps,
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
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return []
        merged_rows = await agent_case_repo.sparse_search(
            terms,
            where,
            columns=AgentCase.BM25_FIELDS,
            limit=limit,
        )
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in merged_rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        rows = await agent_case_repo.dense_search(vector, where, limit=limit)
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]
