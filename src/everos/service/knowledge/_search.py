"""The knowledge search use case: recall, category-aware rerank, hit shaping.

Recall over the knowledge-topic index feeds everalgo's category-aware
retrieval: the recall pool's category mass is rolled up, the pool is
reranked on the topics' full content (read from the catalog), and a small
category boost breaks ties. Search cannot run without both an embedding and
a rerank provider; :meth:`KnowledgeSearcher.require_providers` is the one
place that says so.
"""

from __future__ import annotations

__all__ = ["KnowledgeSearcher"]

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from everalgo.rank import acategory_retrieve

from everos.core.errors import ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.memory.models import KnowledgeDocument, KnowledgeTopic
from everos.memory.ports import KnowledgeCatalog
from everos.memory.search.callbacks import build_fetched_rerank_fn

from ._models import (
    DocumentContext,
    SearchHit,
    SearchKnowledgeResult,
    compile_knowledge_where,
)

if TYPE_CHECKING:
    from everos.component.embedding import EmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.config.settings import KnowledgeSearchSettings
    from everos.memory.search.recall import KnowledgeMethod, KnowledgeTopicRecaller

logger = get_logger("everos.service.knowledge")

_KNOWLEDGE_FEATURE = "knowledge"


class KnowledgeSearcher:
    """Searches knowledge topics by keyword, vector, or hybrid retrieval.

    Args:
        catalog: Supplies the topics' full content and their documents.
        embedding: Query embedder; ``None`` when not configured.
        reranker: Reranker; ``None`` when not configured.
        recaller: Recall over the knowledge-topic index.
        config: The ``[knowledge.search]`` section.
    """

    def __init__(
        self,
        *,
        catalog: KnowledgeCatalog,
        embedding: EmbeddingProvider | None,
        reranker: RerankProvider | None,
        recaller: KnowledgeTopicRecaller,
        config: KnowledgeSearchSettings,
    ) -> None:
        self._catalog = catalog
        self._embedding = embedding
        self._reranker = reranker
        self._recaller = recaller
        self._config = config

    def require_providers(self) -> tuple[EmbeddingProvider, RerankProvider]:
        """The embedding + rerank providers, which search cannot run without.

        Raises:
            ProviderNotConfiguredError: Embedding (checked first) or rerank
                is not configured.
        """
        if self._embedding is None:
            raise ProviderNotConfiguredError(
                provider="embedding", feature=_KNOWLEDGE_FEATURE
            )
        if self._reranker is None:
            raise ProviderNotConfiguredError(
                provider="rerank", feature=_KNOWLEDGE_FEATURE
            )
        return self._embedding, self._reranker

    async def search(
        self,
        *,
        query: str,
        method: KnowledgeMethod,
        top_k: int,
        score_threshold: float | None,
        include_content: bool,
        app_id: str,
        project_id: str,
    ) -> SearchKnowledgeResult:
        """Rank the project's topics for ``query``."""
        t0 = time.monotonic()
        where = compile_knowledge_where(app_id, project_id)
        embedder, reranker = self.require_providers()
        vector = await embedder.embed(query)

        retriever = self._recaller.retriever(where, method=method, vector=vector)
        contents: dict[str, str] = {}

        async def passages(topic_ids: Sequence[str]) -> list[str]:
            topics = await self._catalog.get_topics(topic_ids)
            contents.update({t.topic_id: t.content for t in topics})
            return [contents.get(tid, "") for tid in topic_ids]

        config = self._config
        ranked = await acategory_retrieve(
            query,
            base_retrieve=retriever,
            rerank_fn=build_fetched_rerank_fn(reranker, passages=passages),
            recall_n=config.recall_n,
            rerank_n=config.rerank_n,
            mass_top_m=config.mass_top_m,
            lam=config.lam,
            top_n=min(top_k, config.top_k_cap),
        )
        if score_threshold is not None:
            ranked = [c for c in ranked if c.score >= score_threshold]

        documents = {
            d.doc_id: d
            for d in await self._catalog.get_documents(
                {retriever.topics_by_id[c.id].doc_id for c in ranked}
            )
        }
        hits = [
            _search_hit(
                retriever.topics_by_id[c.id],
                score=c.score,
                source=None if c.source == "other" else c.source,
                method=method,
                document=documents.get(retriever.topics_by_id[c.id].doc_id),
                content=contents.get(c.id, "") if include_content else None,
            )
            for c in ranked
        ]

        took_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "knowledge_search_complete",
            method=method,
            query_len=len(query),
            hits=len(hits),
            took_ms=round(took_ms, 1),
        )
        return SearchKnowledgeResult(hits=hits, total=len(hits), took_ms=took_ms)


def _search_hit(
    topic: KnowledgeTopic,
    *,
    score: float,
    source: str | None,
    method: str,
    document: KnowledgeDocument | None,
    content: str | None,
) -> SearchHit:
    return SearchHit(
        topic_id=topic.topic_id,
        category_id=topic.category_id,
        topic_name=topic.topic_name,
        topic_path=topic.topic_path,
        depth=topic.depth,
        summary=topic.summary,
        content=content,
        score=score,
        retrieval_method=method,
        source=source,
        document=DocumentContext(
            doc_id=document.doc_id if document else "",
            title=document.title if document else "",
            summary=document.summary if document else "",
        ),
    )
