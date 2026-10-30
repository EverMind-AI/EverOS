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

from everos.memory.models import AgentCase
from everos.memory.ports import AgentCaseIndex, Filter

from .base import RecallerDeps, RecallResult, from_hits


class AgentCaseRecaller:
    """BM25 (dual-column) + vector recall over the agent-case index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The agent-case index port.
    """

    kind: ClassVar[str] = "agent_case"
    everalgo_memory_type: ClassVar[str] = "case"

    def __init__(self, deps: RecallerDeps, index: AgentCaseIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[AgentCase]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token doesn't poison the column query (see
        ``EpisodeRecaller.sparse_recall``). One BooleanQuery is built
        per BM25 column (``MatchQuery`` is column-bound), then the
        two per-column result lists merge by id keeping the max score.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return from_hits(hits, source="keyword")

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[AgentCase]:
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return from_hits(hits, source="vector")
