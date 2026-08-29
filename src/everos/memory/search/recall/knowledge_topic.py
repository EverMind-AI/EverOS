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

from everos.infra.persistence.index import (
    KnowledgeTopic,
    Predicate,
    knowledge_topic_repo,
)

from .base import (
    RecallerDeps,
    cosine_score_from_distance,
    row_to_candidate,
)


def _merge_bm25_results(
    per_column: tuple[list[dict], ...],
    *,
    limit: int,
) -> list[dict]:
    """Merge multi-column BM25 results by id, keeping max score."""
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
    return sorted(
        best.values(),
        key=lambda r: float(r.get("_score", 0.0)),
        reverse=True,
    )[:limit]


class KnowledgeTopicRecaller:
    """BM25 (dual-column) + vector recall over the LanceDB ``knowledge_topic`` table.

    Args:
        deps: Shared recaller dependencies (tokenizer, embedding provider).
    """

    kind: ClassVar[str] = "knowledge_topic"
    everalgo_memory_type: ClassVar[str] = "knowledge"
    text_field: ClassVar[str] = "summary"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: Predicate, *, limit: int
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
        merged_rows = await knowledge_topic_repo.sparse_search(
            terms,
            where,
            columns=KnowledgeTopic.BM25_FIELDS,
            limit=limit,
        )
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in merged_rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: Predicate, *, limit: int
    ) -> list[Candidate]:
        """Cosine ANN over the ``summary`` vector (1024-d)."""
        if not vector:
            return []
        rows = await knowledge_topic_repo.dense_search(vector, where, limit=limit)
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]
