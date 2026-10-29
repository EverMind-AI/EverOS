"""The knowledge read use cases: documents, topics and categories.

Every read is served from the catalog (the records the cascade keeps), so a
document is visible here only once the cascade has indexed it. The store is
consulted for what only the markdown tree holds: the taxonomy and the
uploaded original.
"""

from __future__ import annotations

__all__ = ["KnowledgeQueries"]

import asyncio
from datetime import datetime

from everos.core.errors import DocumentNotFoundError, TopicNotFoundError
from everos.memory.models import KnowledgeTopic
from everos.memory.ports import KnowledgeCatalog, KnowledgeStore, Page

from ._models import (
    CategoryOverview,
    DocumentDetail,
    DocumentListResult,
    DocumentOverviewItem,
    TopicDetail,
    TopicOverview,
)


class KnowledgeQueries:
    """Answers the document / topic / category reads.

    Args:
        store: The markdown truth of documents, topics and the taxonomy.
        catalog: The document / topic records the cascade keeps.
    """

    def __init__(self, *, store: KnowledgeStore, catalog: KnowledgeCatalog) -> None:
        self._store = store
        self._catalog = catalog

    async def get_document(self, doc_id: str) -> DocumentDetail:
        """The document with its topic list and stored original."""
        document = await self._catalog.get_document(doc_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {doc_id!r} not found")
        topics = await self._catalog.topics_of(doc_id)
        original = await self._store.original_file(document)
        return DocumentDetail(
            doc_id=document.doc_id,
            category_id=document.category_id,
            title=document.title,
            summary=document.summary,
            source_name=document.source_name,
            source_type=document.source_type,
            original_file_path=original,
            topics=[_topic_overview(t) for t in topics],
            created_at=_recorded(document.created_at),
            updated_at=_recorded(document.updated_at),
        )

    async def list_documents(
        self,
        app_id: str,
        project_id: str,
        category_id: str | None,
        *,
        page: int,
        page_size: int,
        sort_by: str,
        sort_order: str,
    ) -> DocumentListResult:
        """One page of documents with per-document topic counts."""
        listed = await self._catalog.list_documents(
            app_id,
            project_id,
            category_id=category_id,
            page=Page(
                sort_by=sort_by,
                number=page,
                size=page_size,
                descending=sort_order.lower() != "asc",
            ),
        )
        counts = await asyncio.gather(
            *(self._catalog.count_topics(d.doc_id) for d in listed.items)
        )
        return DocumentListResult(
            documents=[
                DocumentOverviewItem(
                    doc_id=d.doc_id,
                    category_id=d.category_id,
                    title=d.title,
                    topic_count=count,
                    created_at=_recorded(d.created_at),
                )
                for d, count in zip(listed.items, counts, strict=True)
            ],
            total=listed.total,
            page=page,
            page_size=page_size,
        )

    async def get_topic(self, topic_id: str) -> TopicDetail:
        """One topic with its full content."""
        topics = await self._catalog.get_topics([topic_id])
        if not topics:
            raise TopicNotFoundError(f"Topic {topic_id!r} not found")
        t = topics[0]
        return TopicDetail(
            topic_id=t.topic_id,
            doc_id=t.doc_id,
            category_id=t.category_id,
            topic_name=t.topic_name,
            topic_path=t.topic_path,
            depth=t.depth,
            summary=t.summary,
            content=t.content,
            content_labels=t.content_labels,
            parent_topic_id=t.parent_topic_id,
            children_topic_ids=t.children_topic_ids,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )

    async def list_categories(
        self, app_id: str, project_id: str
    ) -> list[CategoryOverview]:
        """The taxonomy (created with its defaults on first use), with counts."""
        categories = await self._store.categories(app_id, project_id)
        counts = await self._catalog.count_by_category(app_id, project_id)
        return [
            CategoryOverview(
                category_id=c.id,
                description=c.description,
                document_count=counts.get(c.id, 0),
            )
            for c in categories
        ]


def _topic_overview(topic: KnowledgeTopic) -> TopicOverview:
    return TopicOverview(
        topic_id=topic.topic_id,
        topic_name=topic.topic_name,
        topic_path=topic.topic_path,
        depth=topic.depth,
        summary=topic.summary,
    )


def _recorded(stamp: datetime | None) -> datetime:
    """A catalog timestamp; the catalog always sets one."""
    if stamp is None:
        raise ValueError("catalog document without a timestamp")
    return stamp
