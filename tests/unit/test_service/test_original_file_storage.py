"""Integration-level tests for original file storage.

Tests the full call chain: create_document → _write_document →
_write_original_file → filesystem, and get_document → _resolve_original_file_path.

Only the LLM extractor is mocked; KnowledgeWriter and filesystem are real.
This catches wiring bugs (wrong path passed between functions) that
isolated unit tests miss.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import CategorySpec, KnowledgeMemory, ParsedContent

from everos.service.knowledge import (
    CategoryOverview,
    DocumentDetail,
    DocumentOverviewItem,
    create_document,
    get_document,
    list_categories,
    replace_document,
)

_MOD = "everos.service.knowledge"
_ORIGINAL_DIR = "_original"


def _make_memories(doc_id: str, category_id: str = "Sports") -> list[KnowledgeMemory]:
    return [
        KnowledgeMemory(
            doc_id=doc_id,
            topic_index=0,
            topic="Root Topic",
            topic_path="Root Topic",
            summary="Root summary.",
            content="",
            depth=0,
            category_id=category_id,
        ),
        KnowledgeMemory(
            doc_id=doc_id,
            topic_index=1,
            topic="Sub Topic",
            topic_path="Root Topic > Sub Topic",
            summary="Sub summary.",
            content="Detailed content here.",
            depth=1,
            parent_index=0,
            children_index=[],
            category_id=category_id,
        ),
    ]


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    d = tmp_path / "knowledge"
    d.mkdir()
    return d


# ── TC-1: create_document full chain writes _original/ ──────────────────────


async def test_create_document_writes_original_file(
    knowledge_dir: Path,
) -> None:
    """Full chain: create_document → KnowledgeWriter.write → _write_original_file.

    Only the extractor is mocked. KnowledgeWriter and filesystem are real.
    This catches path wiring bugs between _write_document and _write_original_file.
    """
    doc_id = "d_test00000001"
    file_content = b"original PDF binary data"
    memories = _make_memories(doc_id)
    mock_ext = AsyncMock()
    mock_ext.aextract.return_value = memories

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
        patch(f"{_MOD}._mint_doc_id", return_value=doc_id),
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        result = await create_document(
            extractor=mock_ext,
            parsed=ParsedContent(text="some content"),
            title="Test Doc",
            knowledge_dir=knowledge_dir,
            source_name="report.pdf",
            source_type="file",
            doc_id=doc_id,
            category_id="Sports",
            file_content=file_content,
        )

    # _original/ must be inside the document directory, not the category directory
    doc_dir = Path(result.md_path)
    original_file = doc_dir / _ORIGINAL_DIR / "report.pdf"
    assert original_file.is_file(), f"Expected {original_file} to exist"
    assert original_file.read_bytes() == file_content

    # Category directory must NOT contain _original/
    category_dir = doc_dir.parent
    assert not (category_dir / _ORIGINAL_DIR).exists(), (
        f"_original/ landed in category dir {category_dir}, not doc dir"
    )


# ── TC-2: create_document without file_content skips _original/ ─────────────


async def test_create_document_without_file_content_no_original(
    knowledge_dir: Path,
) -> None:
    """No file_content → no _original/ directory created."""
    doc_id = "d_test00000002"
    memories = _make_memories(doc_id)
    mock_ext = AsyncMock()
    mock_ext.aextract.return_value = memories

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
        patch(f"{_MOD}._mint_doc_id", return_value=doc_id),
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=False)

        result = await create_document(
            extractor=mock_ext,
            parsed=ParsedContent(text="some content"),
            title="No Original",
            knowledge_dir=knowledge_dir,
            source_name="test.txt",
            source_type="file",
            doc_id=doc_id,
            category_id="Sports",
        )

    doc_dir = Path(result.md_path)
    assert not (doc_dir / _ORIGINAL_DIR).exists()


# ── TC-3: get_document returns original_file_path when file exists ──────────


async def test_get_document_returns_original_file_path(
    knowledge_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_document derives original_file_path from md_path + source_name."""
    from everos.component.utils.datetime import get_utc_now
    from everos.config import load_settings
    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.sqlite.tables.knowledge import (
        KnowledgeDocumentRow,
    )

    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    MemoryRoot._instance = None

    # Set up: doc dir with _original/ and a fake SQLite row
    doc_rel = Path("app/proj/knowledge/Sports/my_doc")
    doc_abs = tmp_path / doc_rel
    doc_abs.mkdir(parents=True)
    original_dir = doc_abs / _ORIGINAL_DIR
    original_dir.mkdir()
    (original_dir / "report.pdf").write_bytes(b"pdf bytes")
    (doc_abs / "index.md").write_text("---\ntype: knowledge_document\n---\n")

    now = get_utc_now()
    row = KnowledgeDocumentRow(
        doc_id="d_test00000003",
        app_id="app",
        project_id="proj",
        category_id="Sports",
        title="Test",
        summary="Summary",
        source_name="report.pdf",
        source_type="file",
        md_path=str(doc_rel / "index.md"),
        created_at=now,
        updated_at=now,
    )

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=row)
        mock_topic_repo.get_topics_by_doc_id = AsyncMock(return_value=[])

        detail = await get_document("d_test00000003", "app", "proj")

    assert isinstance(detail, DocumentDetail)
    assert detail.original_file_path is not None
    assert detail.original_file_path == str(original_dir / "report.pdf")

    load_settings.cache_clear()
    MemoryRoot._instance = None


# ── TC-4: get_document returns None for legacy doc (no _original/) ──────────


async def test_get_document_returns_none_for_legacy_doc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy documents without _original/ get original_file_path=None."""
    from everos.component.utils.datetime import get_utc_now
    from everos.config import load_settings
    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.sqlite.tables.knowledge import (
        KnowledgeDocumentRow,
    )

    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    MemoryRoot._instance = None

    doc_rel = Path("app/proj/knowledge/Sports/legacy_doc")
    doc_abs = tmp_path / doc_rel
    doc_abs.mkdir(parents=True)

    now = get_utc_now()
    row = KnowledgeDocumentRow(
        doc_id="d_legacy000001",
        app_id="app",
        project_id="proj",
        category_id="Sports",
        title="Legacy",
        summary="Old doc",
        source_name="old.pdf",
        source_type="file",
        md_path=str(doc_rel / "index.md"),
        created_at=now,
        updated_at=now,
    )

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_doc_repo,
        patch(f"{_MOD}.knowledge_topic_sqlite_repo") as mock_topic_repo,
    ):
        mock_doc_repo.get_by_doc_id = AsyncMock(return_value=row)
        mock_topic_repo.get_topics_by_doc_id = AsyncMock(return_value=[])

        detail = await get_document("d_legacy000001", "app", "proj")

    assert detail.original_file_path is None

    load_settings.cache_clear()
    MemoryRoot._instance = None


# ── TC-5: replace_document writes new _original/ ────────────────────────────


async def test_replace_document_writes_new_original(
    knowledge_dir: Path,
) -> None:
    """replace_document writes new original file after atomic replacement."""
    from everos.component.utils.datetime import get_utc_now
    from everos.infra.persistence.sqlite.tables.knowledge import (
        KnowledgeDocumentRow,
    )

    doc_id = "d_repl00000001"
    old_memories = _make_memories(doc_id)
    new_memories = _make_memories(doc_id)
    new_file = b"new version binary"

    # Phase 1: create the original document
    mock_ext = AsyncMock()
    mock_ext.aextract.return_value = old_memories

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=False)
        old_result = await create_document(
            extractor=mock_ext,
            parsed=ParsedContent(text="old content"),
            title="Old Doc",
            knowledge_dir=knowledge_dir,
            source_name="v1.pdf",
            doc_id=doc_id,
            category_id="Sports",
            file_content=b"old version binary",
        )

    old_dir = Path(old_result.md_path)
    assert (old_dir / _ORIGINAL_DIR / "v1.pdf").read_bytes() == b"old version binary"

    # Phase 2: replace with new content
    now = get_utc_now()
    existing_row = KnowledgeDocumentRow(
        doc_id=doc_id,
        app_id="default",
        project_id="default",
        category_id="Sports",
        title="Old Doc",
        summary="Old summary",
        source_name="v1.pdf",
        source_type="file",
        md_path=str(old_dir / "index.md"),
        created_at=now,
        updated_at=now,
    )

    mock_ext.aextract.return_value = new_memories
    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.get_by_doc_id = AsyncMock(return_value=existing_row)
        new_result = await replace_document(
            extractor=mock_ext,
            parsed=ParsedContent(text="new content"),
            title="New Doc",
            doc_id=doc_id,
            knowledge_dir=knowledge_dir,
            source_name="v2.pdf",
            category_id="Sports",
            file_content=new_file,
        )

    new_dir = Path(new_result.md_path)
    assert (new_dir / _ORIGINAL_DIR / "v2.pdf").read_bytes() == new_file


# ── TC-6: delete removes _original/ with doc dir ────────────────────────────


async def test_delete_removes_original_with_doc_dir(
    knowledge_dir: Path,
) -> None:
    """rmtree on doc dir clears _original/ naturally."""
    doc_id = "d_del000000001"
    memories = _make_memories(doc_id)
    mock_ext = AsyncMock()
    mock_ext.aextract.return_value = memories

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=False)
        result = await create_document(
            extractor=mock_ext,
            parsed=ParsedContent(text="content"),
            title="To Delete",
            knowledge_dir=knowledge_dir,
            source_name="file.pdf",
            doc_id=doc_id,
            category_id="Sports",
            file_content=b"data",
        )

    doc_dir = Path(result.md_path)
    assert (doc_dir / _ORIGINAL_DIR / "file.pdf").is_file()

    shutil.rmtree(doc_dir)
    assert not list(doc_dir.parent.glob(doc_dir.name))


# ── TC-7: shutil.move preserves _original/ ──────────────────────────────────


async def test_move_preserves_original(knowledge_dir: Path) -> None:
    """PATCH category move (shutil.move) keeps _original/ intact."""
    doc_id = "d_move00000001"
    memories = _make_memories(doc_id)
    file_content = b"important data"
    mock_ext = AsyncMock()
    mock_ext.aextract.return_value = memories

    with (
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.doc_id_exists = AsyncMock(return_value=False)
        result = await create_document(
            extractor=mock_ext,
            parsed=ParsedContent(text="content"),
            title="To Move",
            knowledge_dir=knowledge_dir,
            source_name="file.pdf",
            doc_id=doc_id,
            category_id="Sports",
            file_content=file_content,
        )

    old_dir = Path(result.md_path)
    new_dir = knowledge_dir / "Finance" / old_dir.name
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_dir), str(new_dir))

    assert (new_dir / _ORIGINAL_DIR / "file.pdf").read_bytes() == file_content
    assert not list(old_dir.parent.glob(old_dir.name))


# ── TC-8: DocumentOverviewItem slim fields ──────────────────────────────────


def test_document_overview_item_slim_fields() -> None:
    """DocumentOverviewItem has exactly 5 fields, no summary/source/updated_at."""
    from everos.component.utils.datetime import get_utc_now

    item = DocumentOverviewItem(
        doc_id="d_abc",
        category_id="Tech",
        title="T",
        topic_count=1,
        created_at=get_utc_now(),
    )
    fields = {f.name for f in item.__dataclass_fields__.values()}
    assert fields == {"doc_id", "category_id", "title", "topic_count", "created_at"}


# ── TC-9: list_categories returns document_count ────────────────────────────


async def test_list_categories_document_count() -> None:
    """list_categories merges taxonomy specs with SQLite counts."""
    specs = [
        CategorySpec(id="Tech", description="Technology"),
        CategorySpec(id="Empty", description="No docs"),
    ]
    counts = {"Tech": 3}

    with (
        patch(f"{_MOD}.MemoryRoot"),
        patch(f"{_MOD}.ensure_taxonomy", new_callable=AsyncMock),
        patch(f"{_MOD}.parse_taxonomy", new_callable=AsyncMock, return_value=specs),
        patch(f"{_MOD}.knowledge_document_repo") as mock_repo,
    ):
        mock_repo.count_by_category = AsyncMock(return_value=counts)
        result = await list_categories("app", "proj")

    assert len(result) == 2
    assert all(isinstance(c, CategoryOverview) for c in result)
    assert result[0].document_count == 3
    assert result[1].document_count == 0
