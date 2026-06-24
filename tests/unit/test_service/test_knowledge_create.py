"""Unit tests for :func:`everos.service.knowledge.create_document`.

White-box surfaces: ``KnowledgeExtractor.aextract`` (mocked),
``KnowledgeWriter.write`` (mocked), ``knowledge_document_repo.doc_id_exists``
(mocked), ``ensure_taxonomy`` / ``parse_taxonomy`` (mocked).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import KnowledgeMemory, ParsedContent

from everos.service.knowledge import (
    CreateDocumentResult,
    DuplicateDocumentError,
    ExtractionEmptyError,
    create_document,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_memory(
    doc_id: str = "d_abc123000000",
    topic_index: int = 0,
    category_id: str = "Technology",
) -> KnowledgeMemory:
    return KnowledgeMemory(
        doc_id=doc_id,
        topic_index=topic_index,
        topic=f"Topic {topic_index}",
        topic_path=f"Topic {topic_index}",
        summary=f"Summary for topic {topic_index}",
        content=f"Content for topic {topic_index}",
        depth=0 if topic_index == 0 else 1,
        category_id=category_id,
    )


def _make_memories(
    doc_id: str = "d_abc123000000",
    count: int = 3,
    category_id: str = "Technology",
) -> list[KnowledgeMemory]:
    """Root (index 0) + ``count-1`` topic nodes."""
    return [
        _make_memory(doc_id=doc_id, topic_index=i, category_id=category_id)
        for i in range(count)
    ]


@pytest.fixture
def mock_extractor() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    return tmp_path / "knowledge"


# ── Shared patch targets ─────────────────────────────────────────────────

_MOD = "everos.service.knowledge"


# ── Tests ────────────────────────────────────────────────────────────────


async def test_create_document_success(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """Happy path: extractor returns 3 memories, writer writes, result OK."""
    memories = _make_memories(count=3)
    mock_extractor.aextract.return_value = memories
    doc_dir = knowledge_dir / "Technology" / "test_doc"

    with (
        patch(f"{_MOD}.ensure_taxonomy") as mock_ensure,
        patch(f"{_MOD}.parse_taxonomy", return_value=[]),
        patch(f"{_MOD}.KnowledgeWriter") as mock_writer_cls,
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_writer_cls.write = AsyncMock(return_value=doc_dir)
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        result = await create_document(
            extractor=mock_extractor,
            parsed=ParsedContent(text="Some document content"),
            title="Test Doc",
            knowledge_dir=knowledge_dir,
            doc_id="d_abc123000000",
            source_name="test.pdf",
            source_type="file",
        )

    assert isinstance(result, CreateDocumentResult)
    assert result.doc_id == "d_abc123000000"
    assert result.category_id == "Technology"
    assert result.topic_count == 2  # 3 memories, 1 root (index 0)
    assert result.source_name == "test.pdf"
    assert result.md_path == str(doc_dir)
    mock_ensure.assert_called_once_with(knowledge_dir)
    mock_writer_cls.write.assert_awaited_once()


async def test_create_document_empty_result_raises(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """Extractor returns empty list -> ExtractionEmptyError."""
    mock_extractor.aextract.return_value = []

    with (
        patch(f"{_MOD}.ensure_taxonomy"),
        patch(f"{_MOD}.parse_taxonomy", return_value=[]),
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        with pytest.raises(ExtractionEmptyError):
            await create_document(
                extractor=mock_extractor,
                parsed=ParsedContent(text="Empty doc"),
                title="Empty",
                knowledge_dir=knowledge_dir,
                doc_id="d_abc123000000",
            )


async def test_create_document_mints_doc_id(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """No doc_id provided -> one is minted (starts with 'd_', 14 chars)."""
    memories = _make_memories(count=1, doc_id="d_placeholder0")
    mock_extractor.aextract.return_value = memories
    doc_dir = knowledge_dir / "Technology" / "test_doc"

    with (
        patch(f"{_MOD}.ensure_taxonomy"),
        patch(f"{_MOD}.parse_taxonomy", return_value=[]),
        patch(f"{_MOD}.KnowledgeWriter") as mock_writer_cls,
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_writer_cls.write = AsyncMock(return_value=doc_dir)
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        result = await create_document(
            extractor=mock_extractor,
            parsed=ParsedContent(text="Content"),
            title="Title",
            knowledge_dir=knowledge_dir,
            # doc_id intentionally omitted
        )

    assert result.doc_id.startswith("d_")
    assert len(result.doc_id) == 14  # "d_" + 12 hex chars


async def test_create_document_uses_provided_doc_id(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """Explicit doc_id='d_existing123' -> passed to extractor, not minted."""
    provided_id = "d_existing123"
    memories = _make_memories(count=1, doc_id=provided_id)
    mock_extractor.aextract.return_value = memories
    doc_dir = knowledge_dir / "Technology" / "test_doc"

    with (
        patch(f"{_MOD}.ensure_taxonomy"),
        patch(f"{_MOD}.parse_taxonomy", return_value=[]),
        patch(f"{_MOD}.KnowledgeWriter") as mock_writer_cls,
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_writer_cls.write = AsyncMock(return_value=doc_dir)
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        result = await create_document(
            extractor=mock_extractor,
            parsed=ParsedContent(text="Content"),
            title="Title",
            knowledge_dir=knowledge_dir,
            doc_id=provided_id,
        )

    assert result.doc_id == provided_id
    # Verify the extractor received the provided doc_id.
    call_kwargs = mock_extractor.aextract.call_args
    assert call_kwargs.kwargs["doc_id"] == provided_id


async def test_create_document_empty_category_fallback(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """Memories with empty category_id -> writer receives 'Others'."""
    memories = _make_memories(count=2, category_id="")
    mock_extractor.aextract.return_value = memories
    doc_dir = knowledge_dir / "Others" / "test_doc"

    with (
        patch(f"{_MOD}.ensure_taxonomy"),
        patch(f"{_MOD}.parse_taxonomy", return_value=[]),
        patch(f"{_MOD}.KnowledgeWriter") as mock_writer_cls,
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_writer_cls.write = AsyncMock(return_value=doc_dir)
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        result = await create_document(
            extractor=mock_extractor,
            parsed=ParsedContent(text="Content"),
            title="Title",
            knowledge_dir=knowledge_dir,
            doc_id="d_abc123000000",
        )

    assert result.category_id == "Others"
    # Verify the memories passed to writer have the fallback category.
    write_call = mock_writer_cls.write.call_args
    written_memories = write_call.args[0]
    for m in written_memories:
        assert m.category_id == "Others"


async def test_create_document_rejects_duplicate_doc_id(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """Providing an already-existing doc_id raises DuplicateDocumentError."""
    with (
        patch(f"{_MOD}.ensure_taxonomy"),
        patch(f"{_MOD}.parse_taxonomy", return_value=[]),
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=True)

        with pytest.raises(DuplicateDocumentError):
            await create_document(
                extractor=mock_extractor,
                parsed=ParsedContent(text="Content"),
                title="Title",
                knowledge_dir=knowledge_dir,
                doc_id="d_existing123",
            )


async def test_create_document_ensures_taxonomy(
    mock_extractor: AsyncMock,
    knowledge_dir: Path,
) -> None:
    """ensure_taxonomy called with knowledge_dir."""
    memories = _make_memories(count=1)
    mock_extractor.aextract.return_value = memories
    doc_dir = knowledge_dir / "Technology" / "test_doc"

    with (
        patch(f"{_MOD}.ensure_taxonomy") as mock_ensure,
        patch(f"{_MOD}.parse_taxonomy", return_value=[]) as mock_parse,
        patch(f"{_MOD}.KnowledgeWriter") as mock_writer_cls,
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_writer_cls.write = AsyncMock(return_value=doc_dir)
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        await create_document(
            extractor=mock_extractor,
            parsed=ParsedContent(text="Content"),
            title="Title",
            knowledge_dir=knowledge_dir,
            doc_id="d_abc123000000",
        )

    mock_ensure.assert_called_once_with(knowledge_dir)
    mock_parse.assert_called_once_with(knowledge_dir / ".taxonomy.md")
