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

import asyncio
import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, Literal

from everalgo.rank.fusion import rrf
from everalgo.types import Candidate

from everos.memory.models import KnowledgeTopic
from everos.memory.ports import Filter, KnowledgeTopicIndex, ScoredKnowledgeTopic

from .base import CandidateSource, RecallerDeps


@dataclasses.dataclass(frozen=True)
class TopicRecall:
    """Knowledge-topic candidates plus the topics they stand for.

    Topics have no owner, so they travel as plain :class:`KnowledgeTopic`
    rather than as ``Stored`` records.

    Attributes:
        candidates: Ranked recall candidates, keyed by topic id.
        topics_by_id: The topic behind each ``Candidate.id``.
    """

    candidates: list[Candidate]
    topics_by_id: Mapping[str, KnowledgeTopic]


class KnowledgeTopicRecaller:
    """BM25 (dual-column) + vector recall over the knowledge-topic index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The knowledge-topic index port.
    """

    kind: ClassVar[str] = "knowledge_topic"

    def __init__(self, deps: RecallerDeps, index: KnowledgeTopicIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> TopicRecall:
        """Dual-column BM25 recall via OR-mode BooleanQuery per column.

        Queries ``summary_tokens`` (primary) and ``content_tokens``
        (secondary) in parallel. Results merge by id, keeping the max
        BM25 score across the two columns. This ensures that a topic
        matching the query in either its summary or its content body is
        surfaced without double-counting.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return TopicRecall(candidates=[], topics_by_id={})
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return _recall(hits, source="keyword")

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> TopicRecall:
        """Cosine ANN over the ``summary`` vector."""
        if not vector:
            return TopicRecall(candidates=[], topics_by_id={})
        hits = await self._index.dense_search(vector, where, limit=limit)
        return _recall(hits, source="vector")

    def retriever(
        self, where: Filter, *, method: KnowledgeMethod, vector: Sequence[float]
    ) -> TopicRetriever:
        """A ``RetrieveFn`` for one search, recording the topics it returns."""
        return TopicRetriever(self, where, method=method, vector=vector)


type KnowledgeMethod = Literal["keyword", "vector", "hybrid"]


class TopicRetriever:
    """An everalgo ``RetrieveFn`` over the knowledge-topic index.

    ``keyword`` / ``vector`` recall one route; ``hybrid`` recalls both in
    parallel and fuses them with RRF. Every topic it returns is kept in
    :attr:`topics_by_id`, so a caller can turn ranked candidate ids back
    into topics without reading candidate metadata.

    Args:
        recaller: The knowledge-topic recaller.
        where: Filter scoping every recall.
        method: Which route(s) to recall.
        vector: The query embedding (unused by ``keyword``).
    """

    def __init__(
        self,
        recaller: KnowledgeTopicRecaller,
        where: Filter,
        *,
        method: KnowledgeMethod,
        vector: Sequence[float],
    ) -> None:
        self._recaller = recaller
        self._where = where
        self._method = method
        self._vector = vector
        self.topics_by_id: dict[str, KnowledgeTopic] = {}

    async def __call__(self, query: str, limit: int) -> list[Candidate]:
        recalls = await self._recall(query, limit)
        for recall in recalls:
            self.topics_by_id.update(recall.topics_by_id)
        if len(recalls) == 1:
            return recalls[0].candidates
        sparse, dense = recalls
        return rrf(sparse.candidates, dense.candidates)[:limit]

    async def _recall(self, query: str, limit: int) -> list[TopicRecall]:
        if self._method == "keyword":
            return [await self._sparse(query, limit)]
        if self._method == "vector":
            return [await self._dense(limit)]
        return list(
            await asyncio.gather(self._sparse(query, limit), self._dense(limit))
        )

    async def _sparse(self, query: str, limit: int) -> TopicRecall:
        return await self._recaller.sparse_recall(query, self._where, limit=limit)

    async def _dense(self, limit: int) -> TopicRecall:
        return await self._recaller.dense_recall(self._vector, self._where, limit=limit)


def _recall(
    hits: Iterable[ScoredKnowledgeTopic], *, source: CandidateSource
) -> TopicRecall:
    candidates: list[Candidate] = []
    topics_by_id: dict[str, KnowledgeTopic] = {}
    for hit in hits:
        topic = hit.topic
        candidates.append(
            Candidate(
                id=topic.topic_id,
                score=0.0 if hit.score is None else float(hit.score),
                source=source,
                metadata=topic.model_dump(mode="python"),
            )
        )
        topics_by_id[topic.topic_id] = topic
    return TopicRecall(candidates=candidates, topics_by_id=topics_by_id)
