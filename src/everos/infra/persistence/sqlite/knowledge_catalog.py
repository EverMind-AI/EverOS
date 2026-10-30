"""SQLite adapter for the knowledge catalog port.

Wraps the ``knowledge_documents`` / ``knowledge_topics`` repositories (which
the cascade fills from the markdown tree) and owns the row <-> domain
conversion: the JSON-encoded list columns are decoded here, so the domain
sees ``list[str]``. ``md_path`` is a storage column only (the cascade deletes
rows by it); it is written from the store's ``WrittenDocument`` and never
surfaces on the domain document.
"""

from __future__ import annotations

__all__ = ["SqliteKnowledgeCatalog"]

import json
from collections.abc import Collection, Sequence

from everos.memory.models import KnowledgeDocument, KnowledgeTopic
from everos.memory.ports import Page, PageResult

from ._repos import KnowledgeDocumentRepo, KnowledgeTopicRowRepo
from ._repos.knowledge import DocumentUpsertPayload
from ._tables.knowledge import KnowledgeDocumentRow, KnowledgeTopicRow


class SqliteKnowledgeCatalog:
    """Knowledge document and topic records over their two tables.

    Args:
        documents: The ``knowledge_documents`` repository.
        topics: The ``knowledge_topics`` repository.
    """

    def __init__(
        self, documents: KnowledgeDocumentRepo, topics: KnowledgeTopicRowRepo
    ) -> None:
        self._documents = documents
        self._topics = topics

    async def get_document(self, doc_id: str) -> KnowledgeDocument | None:
        row = await self._documents.get_by_doc_id(doc_id)
        return None if row is None else _document(row)

    async def get_documents(self, doc_ids: Collection[str]) -> list[KnowledgeDocument]:
        rows = await self._documents.get_documents_by_ids(set(doc_ids))
        return [_document(r) for r in rows]

    async def document_exists(self, doc_id: str) -> bool:
        return await self._documents.doc_id_exists(doc_id)

    async def list_documents(
        self,
        app_id: str,
        project_id: str,
        *,
        category_id: str | None,
        page: Page,
    ) -> PageResult[KnowledgeDocument]:
        listed = await self._documents.list_documents(
            app_id=app_id,
            project_id=project_id,
            category_id=category_id,
            page=page.number,
            page_size=page.size,
            sort_by=page.sort_by,
            sort_order="desc" if page.descending else "asc",
        )
        return PageResult(items=[_document(r) for r in listed.rows], total=listed.total)

    async def count_by_category(self, app_id: str, project_id: str) -> dict[str, int]:
        return await self._documents.count_by_category(app_id, project_id)

    async def record_document(
        self, document: KnowledgeDocument, *, md_path: str
    ) -> None:
        await self._documents.upsert_from_handler(
            DocumentUpsertPayload(
                doc_id=document.doc_id,
                app_id=document.app_id,
                project_id=document.project_id,
                category_id=document.category_id,
                title=document.title,
                summary=document.summary,
                source_name=document.source_name,
                source_type=document.source_type,
                md_path=md_path,
            )
        )

    async def get_topics(self, topic_ids: Sequence[str]) -> list[KnowledgeTopic]:
        return [
            _topic(r) for r in await self._topics.get_topics_by_ids(list(topic_ids))
        ]

    async def topics_of(self, doc_id: str) -> list[KnowledgeTopic]:
        return [_topic(r) for r in await self._topics.get_topics_by_doc_id(doc_id)]

    async def count_topics(self, doc_id: str) -> int:
        return await self._topics.count_by_doc_id(doc_id)


def _document(row: KnowledgeDocumentRow) -> KnowledgeDocument:
    return KnowledgeDocument(
        doc_id=row.doc_id,
        app_id=row.app_id,
        project_id=row.project_id,
        category_id=row.category_id,
        title=row.title,
        summary=row.summary,
        source_name=row.source_name,
        source_type=row.source_type,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _topic(row: KnowledgeTopicRow) -> KnowledgeTopic:
    return KnowledgeTopic(
        topic_id=row.node_id,
        doc_id=row.doc_id,
        app_id=row.app_id,
        project_id=row.project_id,
        category_id=row.category_id,
        topic_name=row.topic_name,
        topic_path=row.topic_path,
        depth=row.depth,
        summary=row.summary,
        parent_topic_id=row.parent_node_id,
        content=row.content,
        content_labels=_json_list(row.content_labels),
        children_topic_ids=_json_list(row.children_node_ids),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _json_list(text: str | None) -> list[str]:
    return json.loads(text) if text else []
