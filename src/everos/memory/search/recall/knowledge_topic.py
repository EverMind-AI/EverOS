"""KnowledgeTopic recaller — dual-column BM25 + cosine ANN.

The schema declares two BM25 columns (``summary_tokens`` — primary anchor —
and ``content_tokens`` — secondary detail match). LanceDB's
``nearest_to_text`` searches one column at a time, so we run the BM25 query
twice in parallel and merge by row id keeping the max score across columns.
Vector recall is single-shot over the ``summary`` embedding.

Mirrors :class:`AgentCaseRecaller` structurally — both kinds share the
multi-BM25-column pattern.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.memory.ports import Filter, KnowledgeTopicIndex

from .base import (
    RecallerDeps,
    cosine_score_from_distance,
    row_to_candidate,
)


class KnowledgeTopicRecaller:
    """BM25 (dual-column) + vector recall over the knowledge-topic index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The knowledge-topic index port.
    """

    kind: ClassVar[str] = "knowledge_topic"
    everalgo_memory_type: ClassVar[str] = "knowledge"
    text_field: ClassVar[str] = "summary"

    def __init__(self, deps: RecallerDeps, index: KnowledgeTopicIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> list[Candidate]:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Queries ``summary_tokens`` (primary) and ``content_tokens``
        (secondary) in parallel. Results merge by id, keeping the max
        BM25 score across the two columns. This ensures that a topic
        matching the query in either its summary or its content body is
        surfaced without double-counting.
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
        """Cosine ANN over the ``summary`` vector (1024-d)."""
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
