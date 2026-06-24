"""Unit tests for knowledge CRUD service functions.

White-box surfaces mocked:
    ``knowledge_document_repo`` (get_by_doc_id, list_documents, upsert_from_handler)
    ``knowledge_topic_sqlite_repo`` (get_topics_by_doc_id, get_topics_by_ids,
                                     count_by_doc_id)
    ``anyio.Path.is_dir`` + ``anyio.to_thread.run_sync`` for delete_document.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from everos.component.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite.repos.knowledge import DocumentListPage
from everos.infra.persistence.sqlite.tables.knowledge import (
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
)
from everos.service.knowledge import (
    DeleteResult,
    DocumentDetail,
    DocumentListResult,
    DocumentNotFoundError,
    PatchResult,
    TopicDetail,
    TopicNotFoundError,
    delete_document,
    get_document,
    get_topic,
    list_documents,
    patch_document,
)

_MOD = "everos.service.knowledge"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc_row(
    doc_id: str = "d_testdoc00001",
    category_id: str = "Technology",
    title: str = "Test Doc",
    summary: str = "A test document.",
    source_name: str | None = "test.pdf",
    source_type: str | None = "file",
    md_path: str = "/tmp/knowledge/Technology/d_testdoc00001",
) -> KnowledgeDocumentRow:
    now = get_utc_now()
    return KnowledgeDocumentRow(
        doc_id=doc_id,
        app_id="app1",
        project_id="proj1",
        category_id=category_id,
        title=title,
        summary=summary,
        source_name=source_name,
        source_type=source_type,
        md_path=md_path,
        created_at=now,
        updated_at=now,
    )


def _topic_row(
    node_id: str = "n_topic0001",
    doc_id: str = "d_testdoc00001",
    topic_index: int = 1,
    topic_name: str = "Introduction",
    topic_path: str = "Test Doc > Introduction",
    depth: int = 1,
    parent_node_id: str | None = None,
    children_node_ids: str | None = None,
    content_labels: str | None = None,
) -> KnowledgeTopicRow:
    now = get_utc_now()
    return KnowledgeTopicRow(
        node_id=node_id,
        doc_id=doc_id,
        app_id="app1",
        project_id="proj1",
        category_id="Technology",
        topic_index=topic_index,
        topic_name=topic_name,
        topic_path=topic_path,
        depth=depth,
        parent_node_id=parent_node_id,
        children_node_ids=children_node_ids,
        summary="Intro summary.",
        content="Intro content.",
        content_labels=content_labels,
        md_path="/tmp/knowledge/Technology/d_testdoc00001",
        created_at=now,
        updated_at=now,
    )


# ── get_document ──────────────────────────────────────────────────────────────


async def test_get_document_success() -> None:
    """Returns DocumentDetail with topics mapped from node_id → topic_id."""
    doc = _doc_row()
    topics = [
        _topic_row(node_id="n_001", topic_index=1),
        _topic_row(node_id="n_002", topic_index=2, topic_name="Background"),
    ]

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=doc)
        mock_topic_repo.get_topics_by_doc_id = AsyncMock(return_value=topics)

        result = await get_document("d_testdoc00001", "app1", "proj1")

    assert isinstance(result, DocumentDetail)
    assert result.doc_id == "d_testdoc00001"
    assert result.category_id == "Technology"
    assert result.title == "Test Doc"
    assert len(result.topics) == 2
    assert result.topics[0].topic_id == "n_001"
    assert result.topics[1].topic_id == "n_002"
    assert result.topics[1].topic_name == "Background"
    mock_doc_repo.get_by_doc_id.assert_awaited_once_with("d_testdoc00001")
    mock_topic_repo.get_topics_by_doc_id.assert_awaited_once_with("d_testdoc00001")


async def test_get_document_not_found() -> None:
    """Raises DocumentNotFoundError when the doc_id does not exist."""
    with patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo:
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=None)

        with pytest.raises(DocumentNotFoundError):
            await get_document("d_missing", "app1", "proj1")


# ── get_topic ─────────────────────────────────────────────────────────────────


async def test_get_topic_success() -> None:
    """Returns TopicDetail with parsed JSON children_node_ids and content_labels."""
    children = ["n_child1", "n_child2"]
    labels = ["concept", "definition"]
    topic = _topic_row(
        node_id="n_001",
        parent_node_id="n_root",
        children_node_ids=json.dumps(children),
        content_labels=json.dumps(labels),
    )

    with patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo:
        mock_topic_repo.get_topics_by_ids = AsyncMock(return_value=[topic])

        result = await get_topic("n_001", "app1", "proj1")

    assert isinstance(result, TopicDetail)
    assert result.topic_id == "n_001"
    assert result.parent_topic_id == "n_root"
    assert result.children_topic_ids == children
    assert result.content_labels == labels
    mock_topic_repo.get_topics_by_ids.assert_awaited_once_with(["n_001"])


async def test_get_topic_empty_json_fields() -> None:
    """Returns empty lists when children_node_ids and content_labels are None."""
    topic = _topic_row(node_id="n_leaf", children_node_ids=None, content_labels=None)

    with patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo:
        mock_topic_repo.get_topics_by_ids = AsyncMock(return_value=[topic])

        result = await get_topic("n_leaf", "app1", "proj1")

    assert result.children_topic_ids == []
    assert result.content_labels == []
    assert result.parent_topic_id is None


async def test_get_topic_not_found() -> None:
    """Raises TopicNotFoundError when topic_id does not exist."""
    with patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo:
        mock_topic_repo.get_topics_by_ids = AsyncMock(return_value=[])

        with pytest.raises(TopicNotFoundError):
            await get_topic("n_missing", "app1", "proj1")


# ── delete_document ───────────────────────────────────────────────────────────


async def test_delete_document_success(tmp_path: Path) -> None:
    """Calls rmtree on the document directory and returns topic count."""
    doc_dir = tmp_path / "d_testdoc00001"
    doc_dir.mkdir()
    doc = _doc_row(md_path=str(doc_dir))

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo,
        patch(f"{_MOD}.anyio") as mock_anyio,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=doc)
        mock_topic_repo.count_by_doc_id = AsyncMock(return_value=3)

        mock_path_instance = AsyncMock()
        mock_path_instance.is_dir = AsyncMock(return_value=True)
        mock_anyio.Path.return_value = mock_path_instance
        mock_anyio.to_thread.run_sync = AsyncMock(return_value=None)

        result = await delete_document("d_testdoc00001", "app1", "proj1")

    assert isinstance(result, DeleteResult)
    assert result.doc_id == "d_testdoc00001"
    assert result.deleted_topics == 3
    mock_anyio.to_thread.run_sync.assert_awaited_once()


async def test_delete_document_idempotent() -> None:
    """Returns deleted_topics=0 without error when document does not exist."""
    with patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo:
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=None)

        result = await delete_document("d_missing", "app1", "proj1")

    assert isinstance(result, DeleteResult)
    assert result.doc_id == "d_missing"
    assert result.deleted_topics == 0


# ── list_documents ────────────────────────────────────────────────────────────


async def test_list_documents_returns_paginated_result() -> None:
    """Returns DocumentListResult with correct pagination metadata."""
    rows = [
        _doc_row(doc_id="d_doc1", title="Alpha"),
        _doc_row(doc_id="d_doc2", title="Beta"),
    ]
    page_result = DocumentListPage(rows=rows, total=10)

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo,
    ):
        mock_doc_repo.list_documents = AsyncMock(return_value=page_result)
        mock_topic_repo.count_by_doc_id = AsyncMock(return_value=4)

        result = await list_documents(
            "app1",
            "proj1",
            category_id=None,
            page=2,
            page_size=2,
            sort_by="title",
            sort_order="asc",
        )

    assert isinstance(result, DocumentListResult)
    assert result.total == 10
    assert result.page == 2
    assert result.page_size == 2
    assert len(result.documents) == 2
    assert result.documents[0].doc_id == "d_doc1"
    assert result.documents[0].topic_count == 4
    assert result.documents[1].doc_id == "d_doc2"
    mock_doc_repo.list_documents.assert_awaited_once_with(
        app_id="app1",
        project_id="proj1",
        category_id=None,
        page=2,
        page_size=2,
        sort_by="title",
        sort_order="asc",
    )


# ── patch_document ────────────────────────────────────────────────────────────


async def test_patch_document_title_updates_correctly() -> None:
    """Returns PatchResult with updated_fields=['title'] on title change."""
    doc = _doc_row(title="Old Title")

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}._update_index_frontmatter", new_callable=AsyncMock),
        patch(f"{_MOD}.get_utc_now") as mock_now,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=doc)
        mock_doc_repo.upsert_from_handler = AsyncMock(return_value=None)
        fixed_now = get_utc_now()
        mock_now.return_value = fixed_now

        result = await patch_document(
            "d_testdoc00001", "app1", "proj1", title="New Title"
        )

    assert isinstance(result, PatchResult)
    assert result.doc_id == "d_testdoc00001"
    assert "title" in result.updated_fields
    assert "category_id" not in result.updated_fields
    assert result.updated_at == fixed_now
    mock_doc_repo.upsert_from_handler.assert_awaited_once()


async def test_patch_document_no_changes_returns_empty_fields() -> None:
    """Returns PatchResult with empty updated_fields when nothing changed."""
    doc = _doc_row(title="Same Title", category_id="Technology")

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}.get_utc_now") as mock_now,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=doc)
        mock_doc_repo.upsert_from_handler = AsyncMock(return_value=None)
        mock_now.return_value = get_utc_now()

        result = await patch_document(
            "d_testdoc00001",
            "app1",
            "proj1",
            title="Same Title",
            category_id="Technology",
        )

    assert result.updated_fields == []
    mock_doc_repo.upsert_from_handler.assert_not_awaited()


async def test_patch_document_not_found_raises() -> None:
    """Raises DocumentNotFoundError when doc_id does not exist in SQLite or md."""
    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}._locate_index_md", new_callable=AsyncMock) as mock_locate,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=None)
        mock_locate.return_value = None

        with pytest.raises(DocumentNotFoundError):
            await patch_document("d_missing", "app1", "proj1", title="New")
