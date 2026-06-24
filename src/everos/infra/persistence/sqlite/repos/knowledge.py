"""Repositories for ``knowledge_documents`` and ``knowledge_topics`` tables.

Two singleton repos — one per table — wired to the process-wide SQLite engine.
Callers construct rows and pass them in; these repos handle persistence only.
"""

from __future__ import annotations

import dataclasses

from sqlalchemy import asc, delete, desc, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.component.utils.datetime import get_utc_now
from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables.knowledge import KnowledgeDocumentRow, KnowledgeTopicRow

_SORTABLE_COLUMNS: dict[str, object] = {
    "created_at": KnowledgeDocumentRow.created_at,
    "updated_at": KnowledgeDocumentRow.updated_at,
    "title": KnowledgeDocumentRow.title,
}


@dataclasses.dataclass(frozen=True)
class DocumentListPage:
    """Result of a paginated document list query."""

    rows: list[KnowledgeDocumentRow]
    total: int


@dataclasses.dataclass(frozen=True)
class DocumentUpsertPayload:
    """Payload for document cascade upsert."""

    doc_id: str
    app_id: str
    project_id: str
    category_id: str
    title: str
    summary: str
    source_name: str | None
    source_type: str | None
    md_path: str


@dataclasses.dataclass(frozen=True)
class TopicUpsertPayload:
    """Payload for topic cascade upsert."""

    node_id: str
    doc_id: str
    app_id: str
    project_id: str
    category_id: str
    topic_index: int
    topic_name: str
    topic_path: str
    depth: int
    parent_node_id: str | None
    children_node_ids: str | None
    summary: str
    content: str
    content_labels: str | None
    md_path: str


def _apply_document_fields(
    row: KnowledgeDocumentRow, payload: DocumentUpsertPayload
) -> None:
    """Copy payload fields onto an existing SQLAlchemy row."""
    row.app_id = payload.app_id
    row.project_id = payload.project_id
    row.category_id = payload.category_id
    row.title = payload.title
    row.summary = payload.summary
    row.source_name = payload.source_name
    row.source_type = payload.source_type
    row.md_path = payload.md_path


class _KnowledgeDocumentRepo(RepoBase[KnowledgeDocumentRow]):
    """SQLite repository for ``knowledge_documents`` table."""

    model = KnowledgeDocumentRow

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def get_by_doc_id(self, doc_id: str) -> KnowledgeDocumentRow | None:
        """Return the document row for ``doc_id``, or ``None`` if absent."""
        async with session_scope(self._factory) as s:
            return await s.get(KnowledgeDocumentRow, doc_id)

    async def get_documents_by_ids(
        self, doc_ids: set[str]
    ) -> list[KnowledgeDocumentRow]:
        """Batch-fetch document rows by primary key set."""
        if not doc_ids:
            return []
        async with session_scope(self._factory) as s:
            stmt = select(KnowledgeDocumentRow).where(
                KnowledgeDocumentRow.doc_id.in_(doc_ids)
            )
            return list((await s.execute(stmt)).scalars().all())

    async def delete_by_md_path(self, md_path: str) -> int:
        """Delete all document rows for ``md_path``; return the deleted count."""
        async with session_scope(self._factory) as s:
            result = await s.execute(
                delete(KnowledgeDocumentRow).where(
                    KnowledgeDocumentRow.md_path == md_path
                )
            )
            await s.commit()
            return int(result.rowcount or 0)

    async def doc_id_exists(self, doc_id: str) -> bool:
        """Return ``True`` if a row with ``doc_id`` exists."""
        async with session_scope(self._factory) as s:
            row = await s.get(KnowledgeDocumentRow, doc_id)
            return row is not None

    async def list_documents(
        self,
        *,
        app_id: str,
        project_id: str,
        category_id: str | None,
        page: int,
        page_size: int,
        sort_by: str,
        sort_order: str,
    ) -> DocumentListPage:
        """Return a paginated, optionally filtered slice of documents.

        Args:
            app_id: Tenant application identifier.
            project_id: Tenant project identifier.
            category_id: When provided, restricts results to this category.
            page: 1-based page number.
            page_size: Maximum rows per page.
            sort_by: Column name — one of ``created_at``, ``updated_at``, ``title``.
            sort_order: ``"asc"`` or ``"desc"``.

        Returns:
            DocumentListPage with matched rows and total count.
        """
        col = _SORTABLE_COLUMNS.get(sort_by, KnowledgeDocumentRow.created_at)
        order_fn = asc if sort_order.lower() == "asc" else desc
        offset = (page - 1) * page_size

        base_filter = [
            KnowledgeDocumentRow.app_id == app_id,
            KnowledgeDocumentRow.project_id == project_id,
        ]
        if category_id is not None:
            base_filter.append(KnowledgeDocumentRow.category_id == category_id)

        async with session_scope(self._factory) as s:
            count_stmt = (
                select(func.count())
                .select_from(KnowledgeDocumentRow)
                .where(*base_filter)
            )
            total = (await s.execute(count_stmt)).scalar_one()

            rows_stmt = (
                select(KnowledgeDocumentRow)
                .where(*base_filter)
                .order_by(order_fn(col))  # type: ignore[arg-type]  -- col is SA column via dict lookup; static type is object
                .offset(offset)
                .limit(page_size)
            )
            rows = list((await s.execute(rows_stmt)).scalars().all())

        return DocumentListPage(rows=rows, total=int(total))

    async def count_by_category(self, app_id: str, project_id: str) -> dict[str, int]:
        """Return ``{category_id: document_count}`` for all categories with docs."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(
                    KnowledgeDocumentRow.category_id,
                    func.count().label("cnt"),
                )
                .where(
                    KnowledgeDocumentRow.app_id == app_id,
                    KnowledgeDocumentRow.project_id == project_id,
                )
                .group_by(KnowledgeDocumentRow.category_id)
            )
            rows = (await s.execute(stmt)).all()
        return {r.category_id: r.cnt for r in rows}

    async def upsert_from_handler(self, payload: DocumentUpsertPayload) -> None:
        """Insert or update a document row from the cascade handler.

        Use SQLite's atomic ``ON CONFLICT`` form rather than a
        load-mutate-commit cycle. Category moves can produce concurrent
        cascade events for the old and new ``index.md`` paths; an ORM row
        loaded before a delete can otherwise raise ``StaleDataError`` on
        flush.
        """
        async with session_scope(self._factory) as s:
            now = get_utc_now()
            values = {
                **dataclasses.asdict(payload),
                "created_at": now,
                "updated_at": now,
            }
            stmt = insert(KnowledgeDocumentRow).values(**values)
            update_fields = {
                key: values[key]
                for key in (
                    "app_id",
                    "project_id",
                    "category_id",
                    "title",
                    "summary",
                    "source_name",
                    "source_type",
                    "md_path",
                    "updated_at",
                )
            }
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["doc_id"],
                    set_=update_fields,
                )
            )
            await s.commit()


def _apply_topic_fields(row: KnowledgeTopicRow, payload: TopicUpsertPayload) -> None:
    """Copy payload fields onto an existing SQLAlchemy row."""
    row.doc_id = payload.doc_id
    row.app_id = payload.app_id
    row.project_id = payload.project_id
    row.category_id = payload.category_id
    row.topic_index = payload.topic_index
    row.topic_name = payload.topic_name
    row.topic_path = payload.topic_path
    row.depth = payload.depth
    row.parent_node_id = payload.parent_node_id
    row.children_node_ids = payload.children_node_ids
    row.summary = payload.summary
    row.content = payload.content
    row.content_labels = payload.content_labels
    row.md_path = payload.md_path


class _KnowledgeTopicRepo(RepoBase[KnowledgeTopicRow]):
    """SQLite repository for ``knowledge_topics`` table."""

    model = KnowledgeTopicRow

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def get_topics_by_ids(self, node_ids: list[str]) -> list[KnowledgeTopicRow]:
        """Batch-fetch topic rows by node id list — preserves caller order."""
        if not node_ids:
            return []
        async with session_scope(self._factory) as s:
            stmt = select(KnowledgeTopicRow).where(
                KnowledgeTopicRow.node_id.in_(node_ids)
            )
            rows = list((await s.execute(stmt)).scalars().all())
        by_id = {r.node_id: r for r in rows}
        return [by_id[nid] for nid in node_ids if nid in by_id]

    async def get_topics_by_doc_id(self, doc_id: str) -> list[KnowledgeTopicRow]:
        """Return all topic rows for ``doc_id``, ordered by ``topic_index``."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(KnowledgeTopicRow)
                .where(KnowledgeTopicRow.doc_id == doc_id)
                .order_by(KnowledgeTopicRow.topic_index)
            )
            return list((await s.execute(stmt)).scalars().all())

    async def count_by_doc_id(self, doc_id: str) -> int:
        """Return the number of topic rows for ``doc_id``."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(func.count())
                .select_from(KnowledgeTopicRow)
                .where(KnowledgeTopicRow.doc_id == doc_id)
            )
            return int((await s.execute(stmt)).scalar_one())

    async def delete_by_md_path(self, md_path: str) -> int:
        """Delete all topic rows for ``md_path``; return the deleted count."""
        async with session_scope(self._factory) as s:
            result = await s.execute(
                delete(KnowledgeTopicRow).where(KnowledgeTopicRow.md_path == md_path)
            )
            await s.commit()
            return int(result.rowcount or 0)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all topic rows for ``doc_id``; return the deleted count."""
        async with session_scope(self._factory) as s:
            result = await s.execute(
                delete(KnowledgeTopicRow).where(KnowledgeTopicRow.doc_id == doc_id)
            )
            await s.commit()
            return int(result.rowcount or 0)

    async def upsert_from_handler(self, payload: TopicUpsertPayload) -> None:
        """Insert or update a topic row from the cascade handler.

        Checks by ``node_id`` (PK). If the row exists, all mutable
        columns are overwritten and ``updated_at`` bumps. If not, a
        fresh row is inserted with both ``created_at`` and
        ``updated_at`` set to now.
        """
        async with session_scope(self._factory) as s:
            existing = await s.get(KnowledgeTopicRow, payload.node_id)
            now = get_utc_now()
            if existing is not None:
                _apply_topic_fields(existing, payload)
                existing.updated_at = now
                s.add(existing)
            else:
                row = KnowledgeTopicRow(
                    **dataclasses.asdict(payload),
                    created_at=now,
                    updated_at=now,
                )
                s.add(row)
            await s.commit()


knowledge_document_repo = _KnowledgeDocumentRepo()
knowledge_topic_sqlite_repo = _KnowledgeTopicRepo()
