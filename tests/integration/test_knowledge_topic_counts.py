"""Document-list counts and query budgets against real SQLite storage."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import Connection, event
from sqlmodel import SQLModel

from everos.infra.persistence.sqlite import (
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
    dispose_engine,
    get_engine,
    get_session_factory,
    knowledge_topic_sqlite_repo,
)
from everos.service.knowledge import list_documents


@pytest.fixture
async def select_statements() -> AsyncIterator[list[str]]:
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    statements: list[str] = []

    def record_select(
        conn: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record_select)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_select)
        await dispose_engine()


async def _seed_documents(
    count: int,
    *,
    app_id: str = "app",
    project_id: str = "project",
    category_id: str = "category",
) -> list[str]:
    doc_ids = []
    async with get_session_factory()() as session:
        for index in range(count):
            doc_id = f"{app_id}_{project_id}_{category_id}_{index}"
            doc_ids.append(doc_id)
            session.add(
                KnowledgeDocumentRow(
                    doc_id=doc_id,
                    app_id=app_id,
                    project_id=project_id,
                    category_id=category_id,
                    title=f"Document {index:03d}",
                    summary="Summary",
                    md_path=f"{doc_id}/index.md",
                )
            )
            for topic_index in range(index % 3):
                session.add(
                    KnowledgeTopicRow(
                        node_id=f"{doc_id}_{topic_index}",
                        doc_id=doc_id,
                        app_id=app_id,
                        project_id=project_id,
                        category_id=category_id,
                        topic_index=topic_index,
                        topic_name="Topic",
                        topic_path="Topic",
                        depth=1,
                        summary="Summary",
                        content="Content",
                        md_path=f"{doc_id}/{topic_index}.md",
                    )
                )
        await session.commit()
    return doc_ids


@pytest.mark.parametrize("document_count", [0, 1, 20, 100])
async def test_list_documents_uses_constant_query_count(
    select_statements: list[str], document_count: int
) -> None:
    doc_ids = await _seed_documents(document_count)
    select_statements.clear()

    result = await list_documents(
        "app", "project", page_size=100, sort_by="title", sort_order="asc"
    )

    assert result.total == document_count
    assert [doc.doc_id for doc in result.documents] == doc_ids
    assert [doc.topic_count for doc in result.documents] == [
        index % 3 for index in range(document_count)
    ]
    assert len(select_statements) == (3 if document_count else 2)


async def test_list_documents_preserves_scope_filter_and_pagination(
    select_statements: list[str],
) -> None:
    doc_ids = await _seed_documents(6)
    await _seed_documents(3, app_id="other-app")
    await _seed_documents(3, project_id="other-project")
    await _seed_documents(3, category_id="other-category")
    select_statements.clear()

    result = await list_documents(
        "app",
        "project",
        "category",
        page=2,
        page_size=3,
        sort_by="title",
        sort_order="asc",
    )

    assert (result.total, result.page, result.page_size) == (6, 2, 3)
    assert [doc.doc_id for doc in result.documents] == doc_ids[3:]
    assert [doc.topic_count for doc in result.documents] == [0, 1, 2]
    assert len(select_statements) == 3

    select_statements.clear()
    empty = await list_documents("app", "project", "category", page=3, page_size=3)
    assert empty.total == 6
    assert empty.documents == []
    assert len(select_statements) == 2


async def test_batch_counts_only_requested_documents(
    select_statements: list[str],
) -> None:
    doc_ids = await _seed_documents(6)
    select_statements.clear()

    counts = await knowledge_topic_sqlite_repo.count_by_doc_ids(
        [doc_ids[0], doc_ids[1], doc_ids[2], doc_ids[1], "missing"]
    )

    assert counts == {doc_ids[1]: 1, doc_ids[2]: 2}
    assert len(select_statements) == 1

    select_statements.clear()
    assert await knowledge_topic_sqlite_repo.count_by_doc_ids([]) == {}
    assert select_statements == []
