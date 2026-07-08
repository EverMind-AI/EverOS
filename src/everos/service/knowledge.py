"""Knowledge document CRUD + search use cases.

Functions:
    create_document   — full upload-document pipeline.
    get_document      — fetch document detail with topic list.
    list_documents    — paginated document listing.
    get_topic         — fetch a single topic with content.
    delete_document   — remove document directory (cascade handles SQLite/LanceDB).
    replace_document  — atomic replace (backup + restore on failure).
    patch_document    — update mutable document metadata fields.
    search_knowledge  — knowledge retrieval pipeline (keyword / vector / hybrid).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import shutil
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import anyio
from everalgo.rank.fusion import rrf
from everalgo.types import Candidate, CategorySpec, KnowledgeMemory, ParsedContent

from everos.component.utils.datetime import get_utc_now
from everos.core.errors import (
    ConfigurationError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    ExtractionEmptyError,
    TopicNotFoundError,
)
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.core.persistence.markdown import dump_frontmatter, parse_frontmatter
from everos.infra.persistence.markdown import (
    KnowledgeWriter,
    ensure_taxonomy,
    parse_taxonomy,
)
from everos.infra.persistence.sqlite import (
    DocumentUpsertPayload,
    knowledge_document_repo,
    knowledge_topic_sqlite_repo,
)

if TYPE_CHECKING:
    from everos.component.embedding import EmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.config.settings import KnowledgeSearchSettings
    from everos.memory.search.recall import KnowledgeTopicRecaller

logger = get_logger(__name__)

_FALLBACK_CATEGORY = "Others"
_DOC_ID_PREFIX = "d_"
_DOC_ID_HEX_LEN = 12
_MAX_MINT_RETRIES = 5
_SCOPE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-@+]+$")
_ORIGINAL_DIR_NAME = "_original"


class KnowledgeExtractor(Protocol):
    """Structural type for the algo extractor.

    Matches ``everalgo.knowledge.KnowledgeExtractor.aextract`` so
    callers can pass any compatible implementation (including test
    doubles).
    """

    async def aextract(
        self,
        parsed: ParsedContent,
        *,
        doc_id: str,
        title: str,
        categories: list[CategorySpec] | None = None,
        category_id: str = "",
    ) -> list[KnowledgeMemory]: ...


@dataclasses.dataclass(frozen=True)
class CreateDocumentResult:
    """Returned by :func:`create_document` on success."""

    doc_id: str
    category_id: str
    topic_count: int
    source_name: str | None
    md_path: str
    original_file_path: str | None = None


async def _extract_memories(
    extractor: KnowledgeExtractor,
    parsed: ParsedContent,
    doc_id: str,
    title: str,
    *,
    categories: list[CategorySpec],
    category_id: str | None,
) -> list[KnowledgeMemory]:
    """Run the algo extractor and return non-empty memories.

    Args:
        extractor: Algo extractor injected by the caller.
        parsed: Parsed document content to extract from.
        doc_id: Document identifier passed to the extractor.
        title: Human-readable document title.
        categories: Taxonomy categories for LLM classification.
        category_id: Pre-known category; empty string lets the LLM classify.

    Returns:
        Non-empty list of KnowledgeMemory with category fallback applied.

    Raises:
        ExtractionEmptyError: When the extractor produces no memories.
    """
    memories: list[KnowledgeMemory] = await extractor.aextract(
        parsed,
        doc_id=doc_id,
        title=title,
        categories=categories,
        category_id=category_id or "",
    )
    if not memories:
        raise ExtractionEmptyError(
            f"Extractor returned no memories for doc_id={doc_id!r}"
        )
    return _apply_category_fallback(memories)


async def create_document(
    *,
    extractor: KnowledgeExtractor,
    parsed: ParsedContent,
    title: str,
    knowledge_dir: Path,
    source_name: str | None = None,
    source_type: str | None = None,
    doc_id: str | None = None,
    category_id: str | None = None,
    file_content: bytes | None = None,
) -> CreateDocumentResult:
    """Create a knowledge document from parsed content.

    Args:
        extractor: Algo extractor injected by the caller.
        parsed: Parsed document content (from parser or raw text).
        title: Human-readable document title.
        knowledge_dir: Absolute path from ``MemoryRoot.knowledge_dir``.
        source_name: Optional provenance label (URL, filename, ...).
        source_type: Optional provenance type (``"url"``, ``"file"``, ...).
        doc_id: Caller-provided doc_id; ``None`` mints a new one.
        category_id: Pre-known category; ``None`` lets the LLM classify.
        file_content: Raw uploaded file bytes to persist in ``_original/``.

    Returns:
        Result containing doc_id, category, topic count, and md path.

    Raises:
        DuplicateDocumentError: When ``doc_id`` already exists.
        ExtractionEmptyError: When the extractor produces no memories.
    """
    doc_id = doc_id or await _mint_doc_id()
    if await knowledge_document_repo.doc_id_exists(doc_id):
        raise DuplicateDocumentError(
            f"Document {doc_id!r} already exists; use PUT to replace"
        )
    return await _write_document(
        extractor=extractor,
        parsed=parsed,
        title=title,
        knowledge_dir=knowledge_dir,
        source_name=source_name,
        source_type=source_type,
        doc_id=doc_id,
        category_id=category_id,
        file_content=file_content,
    )


async def _write_document(
    *,
    extractor: KnowledgeExtractor,
    parsed: ParsedContent,
    title: str,
    knowledge_dir: Path,
    source_name: str | None,
    source_type: str | None,
    doc_id: str,
    category_id: str | None,
    file_content: bytes | None = None,
) -> CreateDocumentResult:
    """Extract topics and write markdown — shared by create and replace."""
    await ensure_taxonomy(knowledge_dir)
    categories = await parse_taxonomy(knowledge_dir / ".taxonomy.md")

    memories = await _extract_memories(
        extractor,
        parsed,
        doc_id,
        title,
        categories=categories,
        category_id=category_id,
    )
    resolved_category = memories[0].category_id

    md_path = await KnowledgeWriter.write(
        memories,
        knowledge_dir,
        source_name=source_name,
        source_type=source_type,
    )

    original_file_path: str | None = None
    if file_content and source_name:
        written = await _write_original_file(md_path, source_name, file_content)
        original_file_path = str(written)

    topic_count = sum(1 for m in memories if m.topic_index != 0)

    logger.info(
        "document created",
        doc_id=doc_id,
        category_id=resolved_category,
        topic_count=topic_count,
    )

    return CreateDocumentResult(
        doc_id=doc_id,
        category_id=resolved_category,
        topic_count=topic_count,
        source_name=source_name,
        md_path=str(md_path),
        original_file_path=original_file_path,
    )


# ── CRUD result types ─────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class TopicOverview:
    """Minimal topic summary embedded in :class:`DocumentDetail`."""

    topic_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str


@dataclasses.dataclass(frozen=True)
class DocumentDetail:
    """Full document record returned by :func:`get_document`."""

    doc_id: str
    category_id: str
    title: str
    summary: str
    source_name: str | None
    source_type: str | None
    original_file_path: str | None
    topics: list[TopicOverview]
    created_at: datetime
    updated_at: datetime


@dataclasses.dataclass(frozen=True)
class DocumentOverviewItem:
    """One row in a paginated document list."""

    doc_id: str
    category_id: str
    title: str
    topic_count: int
    created_at: datetime


@dataclasses.dataclass(frozen=True)
class DocumentListResult:
    """Paginated document list returned by :func:`list_documents`."""

    documents: list[DocumentOverviewItem]
    total: int
    page: int
    page_size: int


@dataclasses.dataclass(frozen=True)
class TopicDetail:
    """Full topic record returned by :func:`get_topic`."""

    topic_id: str
    doc_id: str
    category_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str
    content: str
    content_labels: list[str]
    parent_topic_id: str | None
    children_topic_ids: list[str]
    created_at: datetime
    updated_at: datetime


@dataclasses.dataclass(frozen=True)
class CategoryOverview:
    """One taxonomy category with its document count."""

    category_id: str
    description: str
    document_count: int


@dataclasses.dataclass(frozen=True)
class DeleteResult:
    """Result of :func:`delete_document`."""

    doc_id: str
    deleted_topics: int


@dataclasses.dataclass(frozen=True)
class PatchResult:
    """Result of :func:`patch_document`."""

    doc_id: str
    updated_fields: list[str]
    updated_at: datetime


# ── CRUD functions ────────────────────────────────────────────────────────────


async def get_document(
    doc_id: str,
    app_id: str,
    project_id: str,
) -> DocumentDetail:
    """Fetch a document with its topic list.

    Args:
        doc_id: Document primary key.
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.

    Returns:
        DocumentDetail with all topics.

    Raises:
        DocumentNotFoundError: When no row exists for ``doc_id``.
    """
    row = await knowledge_document_repo.get_by_doc_id(doc_id)
    if row is None:
        raise DocumentNotFoundError(f"Document {doc_id!r} not found")

    topic_rows = await knowledge_topic_sqlite_repo.get_topics_by_doc_id(doc_id)
    topics = [
        TopicOverview(
            topic_id=t.node_id,
            topic_name=t.topic_name,
            topic_path=t.topic_path,
            depth=t.depth,
            summary=t.summary,
        )
        for t in topic_rows
    ]

    original_file_path = await _resolve_original_file_path(row.md_path, row.source_name)

    return DocumentDetail(
        doc_id=row.doc_id,
        category_id=row.category_id,
        title=row.title,
        summary=row.summary,
        source_name=row.source_name,
        source_type=row.source_type,
        original_file_path=original_file_path,
        topics=topics,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def list_documents(
    app_id: str,
    project_id: str,
    category_id: str | None = None,
    *,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> DocumentListResult:
    """Return a paginated document list with per-document topic counts.

    Args:
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.
        category_id: Optional category filter.
        page: 1-based page number.
        page_size: Rows per page.
        sort_by: Sort column — ``created_at``, ``updated_at``, or ``title``.
        sort_order: ``"asc"`` or ``"desc"``.

    Returns:
        DocumentListResult with items, total, page, and page_size.
    """
    page_result = await knowledge_document_repo.list_documents(
        app_id=app_id,
        project_id=project_id,
        category_id=category_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    doc_ids = [r.doc_id for r in page_result.rows]
    counts = await asyncio.gather(
        *[knowledge_topic_sqlite_repo.count_by_doc_id(did) for did in doc_ids]
    )
    topic_counts = dict(zip(doc_ids, counts, strict=True))

    items = [
        DocumentOverviewItem(
            doc_id=r.doc_id,
            category_id=r.category_id,
            title=r.title,
            topic_count=topic_counts.get(r.doc_id, 0),
            created_at=r.created_at,
        )
        for r in page_result.rows
    ]

    return DocumentListResult(
        documents=items,
        total=page_result.total,
        page=page,
        page_size=page_size,
    )


async def get_topic(
    topic_id: str,
    app_id: str,
    project_id: str,
) -> TopicDetail:
    """Fetch a single topic with full content.

    Args:
        topic_id: Topic node_id primary key.
        app_id: Tenant application identifier (unused but part of scoped API).
        project_id: Tenant project identifier (unused but part of scoped API).

    Returns:
        TopicDetail with parsed JSON list fields.

    Raises:
        TopicNotFoundError: When no row exists for ``topic_id``.
    """
    rows = await knowledge_topic_sqlite_repo.get_topics_by_ids([topic_id])
    if not rows:
        raise TopicNotFoundError(f"Topic {topic_id!r} not found")

    t = rows[0]
    children = json.loads(t.children_node_ids) if t.children_node_ids else []
    labels = json.loads(t.content_labels) if t.content_labels else []

    return TopicDetail(
        topic_id=t.node_id,
        doc_id=t.doc_id,
        category_id=t.category_id,
        topic_name=t.topic_name,
        topic_path=t.topic_path,
        depth=t.depth,
        summary=t.summary,
        content=t.content,
        content_labels=labels,
        parent_topic_id=t.parent_node_id,
        children_topic_ids=children,
        created_at=t.created_at,
        updated_at=t.updated_at,
    )


async def delete_document(
    doc_id: str,
    app_id: str,
    project_id: str,
) -> DeleteResult:
    """Remove a document directory; cascade handles SQLite/LanceDB cleanup.

    Idempotent: returns ``deleted_topics=0`` when the document does not exist.

    Args:
        doc_id: Document primary key.
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.

    Returns:
        DeleteResult with the topic count that was present before deletion.
    """
    row = await knowledge_document_repo.get_by_doc_id(doc_id)
    if row is None:
        return DeleteResult(doc_id=doc_id, deleted_topics=0)

    topic_count = await knowledge_topic_sqlite_repo.count_by_doc_id(doc_id)

    memory_root = MemoryRoot.default()
    doc_dir = memory_root.root / Path(row.md_path).parent
    if await anyio.Path(doc_dir).is_dir():
        await anyio.to_thread.run_sync(shutil.rmtree, doc_dir)

    logger.info(
        "document deleted",
        doc_id=doc_id,
        topic_count=topic_count,
    )
    return DeleteResult(doc_id=doc_id, deleted_topics=topic_count)


async def replace_document(
    *,
    extractor: KnowledgeExtractor,
    parsed: ParsedContent,
    title: str,
    doc_id: str,
    knowledge_dir: Path,
    source_name: str | None = None,
    source_type: str | None = None,
    category_id: str | None = None,
    file_content: bytes | None = None,
) -> CreateDocumentResult:
    """Replace a document atomically — old data preserved on failure.

    The replacement is performed in-place: the md directory is backed up,
    then ``create_document`` overwrites both md files and SQLite rows via
    upsert.  No explicit SQLite delete happens before the write, so a
    failure during extraction leaves the database intact and the backup
    restore brings the md directory back to its original state.

    Args:
        extractor: Algo extractor injected by the caller.
        parsed: Parsed document content.
        title: Human-readable document title.
        doc_id: Existing document id to replace.
        knowledge_dir: Absolute path from ``MemoryRoot.knowledge_dir``.
        source_name: Optional provenance label.
        source_type: Optional provenance type.
        category_id: Pre-known category or ``None`` for LLM classification.
        file_content: Raw uploaded file bytes to persist in ``_original/``.

    Returns:
        CreateDocumentResult for the new document.

    Raises:
        DocumentNotFoundError: When *doc_id* does not exist.
        ExtractionEmptyError: When extraction produces no memories.
            The original document is restored in this case.
        Exception: All extraction-pipeline errors propagate after the
            backup directory is restored.
    """
    row = await knowledge_document_repo.get_by_doc_id(doc_id)
    if row is None:
        raise DocumentNotFoundError(f"Document {doc_id!r} not found")

    backup = await _backup_doc_dir(doc_id)
    try:
        result = await _write_document(
            extractor=extractor,
            parsed=parsed,
            title=title,
            knowledge_dir=knowledge_dir,
            source_name=source_name,
            source_type=source_type,
            doc_id=doc_id,
            category_id=category_id,
            file_content=file_content,
        )
    except Exception:
        await _restore_backup(backup, doc_id)
        raise

    await _drop_backup(backup)
    return result


async def _backup_doc_dir(doc_id: str) -> tuple[Path, Path] | None:
    """Move the existing document directory to a hidden backup path.

    Returns ``(backup_path, original_path)`` so callers can restore
    without reverse-engineering the name, or ``None`` when no directory
    exists.
    """
    memory_root = MemoryRoot.default()
    row = await knowledge_document_repo.get_by_doc_id(doc_id)
    if row is None:
        return None

    original_dir = memory_root.root / Path(row.md_path).parent
    if not await anyio.Path(original_dir).is_dir():
        return None

    backup_dir = original_dir.with_name(f".{original_dir.name}.backup")
    await anyio.to_thread.run_sync(shutil.move, str(original_dir), str(backup_dir))
    return backup_dir, original_dir


async def _restore_backup(
    backup: tuple[Path, Path] | None,
    doc_id: str,
) -> None:
    """Restore a backup created by :func:`_backup_doc_dir`.

    No-op when ``backup`` is ``None`` or the backup no longer exists.
    """
    if backup is None:
        return
    backup_dir, original_dir = backup
    if not await anyio.Path(backup_dir).is_dir():
        return

    await anyio.to_thread.run_sync(shutil.move, str(backup_dir), str(original_dir))
    logger.info("document_backup_restored", doc_id=doc_id)


async def _drop_backup(backup: tuple[Path, Path] | None) -> None:
    """Remove the backup directory after a successful replacement.

    No-op when ``backup`` is ``None`` or already absent.
    """
    if backup is None:
        return
    backup_dir, _ = backup
    if await anyio.Path(backup_dir).is_dir():
        await anyio.to_thread.run_sync(shutil.rmtree, str(backup_dir))


async def _locate_index_md(
    knowledge_dir: Path,
    doc_id: str,
) -> Path | None:
    """Scan knowledge_dir to find the index.md whose frontmatter doc_id matches.

    Used when the SQLite row is absent (cascade hasn't processed the file yet).
    Returns the absolute index.md path, or None if not found.
    """
    adir = anyio.Path(knowledge_dir)
    if not await adir.exists():
        return None
    async for index_md in adir.rglob("index.md"):
        text = await index_md.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        if fm.get("doc_id") == doc_id:
            # anyio.Path → stdlib Path for callers that need synchronous ops.
            return Path(str(index_md))
    return None


async def _update_index_frontmatter(
    index_path: Path,
    title: str,
    category_id: str,
) -> None:
    """Rewrite index.md frontmatter with updated title/category, preserving body."""
    apath = anyio.Path(index_path)
    text = await apath.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    fm["title"] = title
    fm["category_id"] = category_id
    await apath.write_text(dump_frontmatter(fm) + body, encoding="utf-8")


_DIR_SAFE = re.compile(r"[^\w\-.]", re.UNICODE)


def _safe_category(raw: str) -> str:
    """Sanitize category_id for use as a directory name component."""
    slug = raw.replace(" ", "_")
    slug = _DIR_SAFE.sub("", slug)[:50]
    return slug or "Others"


async def _move_doc_directory(
    memory_root: MemoryRoot,
    old_md_path: str,
    new_category: str,
) -> str:
    """Move document directory to new category folder, return new md_path."""
    old_index = memory_root.root / old_md_path
    old_dir = old_index.parent
    new_dir = old_dir.parent.parent / _safe_category(new_category) / old_dir.name
    await anyio.Path(new_dir.parent).mkdir(parents=True, exist_ok=True)
    await anyio.to_thread.run_sync(shutil.move, str(old_dir), str(new_dir))
    new_index = new_dir / "index.md"
    return str(new_index.relative_to(memory_root.root))


async def _update_topics_category(doc_dir: Path, new_category: str) -> None:
    """Rewrite category_id in all topic md files within doc_dir."""
    entries = await anyio.to_thread.run_sync(lambda: sorted(doc_dir.iterdir()))
    topic_files = [f for f in entries if f.suffix == ".md" and f.name != "index.md"]

    async def _rewrite(path: Path) -> None:
        apath = anyio.Path(path)
        text = await apath.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if fm.get("category_id") == new_category:
            return
        fm["category_id"] = new_category
        await apath.write_text(dump_frontmatter(fm) + body, encoding="utf-8")

    await asyncio.gather(*[_rewrite(f) for f in topic_files])


@dataclasses.dataclass(frozen=True)
class _ResolvedDoc:
    """Snapshot of current document state used by :func:`patch_document`."""

    title: str
    category_id: str
    md_path: str
    app_id: str
    project_id: str
    summary: str
    source_name: str | None
    source_type: str | None


async def _resolve_current_doc(
    doc_id: str,
    app_id: str,
    project_id: str,
    memory_root: MemoryRoot,
) -> _ResolvedDoc:
    """Resolve the authoritative document state for patching.

    Raises:
        DocumentNotFoundError: When neither SQLite nor md contain ``doc_id``.
    """
    row = await knowledge_document_repo.get_by_doc_id(doc_id)
    if row is not None:
        return _ResolvedDoc(
            title=row.title,
            category_id=row.category_id,
            md_path=row.md_path,
            app_id=row.app_id,
            project_id=row.project_id,
            summary=row.summary,
            source_name=row.source_name,
            source_type=row.source_type,
        )

    # Cascade may not have synced yet — scan md as fallback.
    knowledge_dir = memory_root.knowledge_dir(app_id, project_id)
    index_md = await _locate_index_md(knowledge_dir, doc_id)
    if index_md is None:
        raise DocumentNotFoundError(f"Document {doc_id!r} not found")
    raw = await anyio.Path(index_md).read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(raw)
    return _ResolvedDoc(
        title=fm.get("title", ""),
        category_id=fm.get("category_id", ""),
        md_path=str(index_md.relative_to(memory_root.root)),
        app_id=app_id,
        project_id=project_id,
        summary="",
        source_name=None,
        source_type=None,
    )


async def _apply_patch_writes(
    doc_id: str,
    current: _ResolvedDoc,
    new_title: str,
    *,
    new_category: str,
    new_md_path: str,
    memory_root: MemoryRoot,
) -> str:
    """Write md frontmatter, move directory if needed, upsert SQLite.

    Returns the (possibly updated) md_path after a category move.
    """
    index_path = memory_root.root / current.md_path
    await _update_index_frontmatter(index_path, new_title, new_category)

    if new_category != current.category_id:
        new_md_path = await _move_doc_directory(
            memory_root, current.md_path, new_category
        )
        new_doc_dir = memory_root.root / Path(new_md_path).parent
        await _update_topics_category(new_doc_dir, new_category)

    await knowledge_document_repo.upsert_from_handler(
        DocumentUpsertPayload(
            doc_id=doc_id,
            app_id=current.app_id,
            project_id=current.project_id,
            category_id=new_category,
            title=new_title,
            summary=current.summary,
            source_name=current.source_name,
            source_type=current.source_type,
            md_path=new_md_path,
        )
    )
    return new_md_path


async def patch_document(
    doc_id: str,
    app_id: str,
    project_id: str,
    title: str | None = None,
    category_id: str | None = None,
) -> PatchResult:
    """Update mutable document metadata in md (truth) and SQLite (immediate).

    Args:
        doc_id: Document primary key.
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.
        title: New title, or ``None`` to leave unchanged.
        category_id: New category, or ``None`` to leave unchanged.

    Returns:
        PatchResult listing updated fields.

    Raises:
        DocumentNotFoundError: When neither SQLite nor md files contain ``doc_id``.
    """
    memory_root = MemoryRoot.default()
    current = await _resolve_current_doc(doc_id, app_id, project_id, memory_root)

    new_title = title if title is not None else current.title
    new_category = category_id if category_id is not None else current.category_id

    updated_fields: list[str] = []
    if new_title != current.title:
        updated_fields.append("title")
    if new_category != current.category_id:
        updated_fields.append("category_id")

    if not updated_fields:
        return PatchResult(doc_id=doc_id, updated_fields=[], updated_at=get_utc_now())

    await _apply_patch_writes(
        doc_id,
        current,
        new_title,
        new_category=new_category,
        new_md_path=current.md_path,
        memory_root=memory_root,
    )
    now = get_utc_now()
    return PatchResult(doc_id=doc_id, updated_fields=updated_fields, updated_at=now)


async def list_categories(app_id: str, project_id: str) -> list[CategoryOverview]:
    """List taxonomy categories with per-category document counts.

    Creates the taxonomy file if it does not yet exist.

    Args:
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.

    Returns:
        List of CategoryOverview with document counts from SQLite.
    """
    knowledge_dir = MemoryRoot.default().knowledge_dir(app_id, project_id)
    await ensure_taxonomy(knowledge_dir)
    specs = await parse_taxonomy(knowledge_dir / ".taxonomy.md")
    counts = await knowledge_document_repo.count_by_category(app_id, project_id)
    return [
        CategoryOverview(
            category_id=s.id,
            description=s.description,
            document_count=counts.get(s.id, 0),
        )
        for s in specs
    ]


async def _mint_doc_id() -> str:
    """Generate a unique ``d_<hex12>`` document id.

    Retries up to ``_MAX_MINT_RETRIES`` times if the id already exists
    in SQLite (astronomically unlikely with 48-bit hex, but defensive).
    """
    for _ in range(_MAX_MINT_RETRIES):
        candidate = f"{_DOC_ID_PREFIX}{uuid4().hex[:_DOC_ID_HEX_LEN]}"
        if not await knowledge_document_repo.doc_id_exists(candidate):
            return candidate
    # Last-resort: use full uuid hex to avoid infinite loops.
    return f"{_DOC_ID_PREFIX}{uuid4().hex}"


def _apply_category_fallback(
    memories: list[KnowledgeMemory],
) -> list[KnowledgeMemory]:
    """Replace empty ``category_id`` with the fallback category."""
    patched: list[KnowledgeMemory] = []
    for m in memories:
        if not m.category_id:
            m = m.model_copy(update={"category_id": _FALLBACK_CATEGORY})
        patched.append(m)
    return patched


async def _resolve_original_file_path(
    md_path: str, source_name: str | None
) -> str | None:
    """Derive the original file path from md_path and source_name.

    Returns the absolute path string if the file exists on disk,
    ``None`` otherwise (legacy documents or missing source_name).
    """
    if not source_name:
        return None
    memory_root = MemoryRoot.default()
    doc_dir = memory_root.root / Path(md_path).parent
    candidate = doc_dir / _ORIGINAL_DIR_NAME / source_name
    if await anyio.Path(candidate).is_file():
        return str(candidate)
    return None


async def _write_original_file(
    doc_dir: Path, source_name: str, file_content: bytes
) -> Path:
    """Write the uploaded binary to ``_original/`` and return its path."""
    original_dir = doc_dir / _ORIGINAL_DIR_NAME
    await anyio.Path(original_dir).mkdir(parents=True, exist_ok=True)
    target = original_dir / source_name
    await anyio.Path(target).write_bytes(file_content)
    return target


# ── Knowledge search ─────────────────────────────────────────────────────────

# Lazy singleton — mirrors the pattern in service/search.py.
_embedding: EmbeddingProvider | None = None
_embedding_resolved = False


def _get_embedding() -> EmbeddingProvider | None:
    """Build the embedding client on first call. ``None`` when not configured."""
    global _embedding, _embedding_resolved
    if _embedding_resolved:
        return _embedding

    from everos.component.embedding import (  # Deferred: singleton
        build_embedding_provider,
    )
    from everos.config import load_settings  # Deferred: singleton

    cfg = load_settings().embedding
    if not cfg.model or cfg.api_key is None:
        logger.warning(
            "knowledge_embedding_not_configured",
            hint="set [embedding] model / api_key to enable vector / hybrid search",
        )
        _embedding = None
    else:
        _embedding = build_embedding_provider(cfg)
        logger.info("knowledge_embedding_built", model=cfg.model)
    _embedding_resolved = True
    return _embedding


# Lazy singleton — mirrors the pattern for _embedding above.
_recaller: KnowledgeTopicRecaller | None = None
_recaller_resolved = False


def _build_recaller() -> KnowledgeTopicRecaller:
    """Return the shared :class:`KnowledgeTopicRecaller`, building it on first call."""
    global _recaller, _recaller_resolved
    if _recaller_resolved:
        return _recaller  # type: ignore[return-value]  -- guarded by _recaller_resolved

    from everos.component.tokenizer import (  # Deferred: singleton
        build_tokenizer,
    )
    from everos.memory.search.recall import (  # Deferred: singleton
        KnowledgeTopicRecaller,
        RecallerDeps,
    )

    _recaller = KnowledgeTopicRecaller(RecallerDeps(tokenizer=build_tokenizer()))
    _recaller_resolved = True
    return _recaller


# Lazy singleton — mirrors the pattern for _embedding above.
_reranker: RerankProvider | None = None
_reranker_resolved = False


def _get_reranker() -> RerankProvider | None:
    """Build the rerank client on first call. ``None`` when not configured."""
    global _reranker, _reranker_resolved
    if _reranker_resolved:
        return _reranker

    from everos.component.rerank import (  # Deferred: singleton
        build_rerank_provider,
    )
    from everos.config import load_settings  # Deferred: singleton

    cfg = load_settings().rerank
    if not cfg.model or not cfg.base_url:
        _reranker = None
    else:
        _reranker = build_rerank_provider(cfg)
        logger.info("knowledge_reranker_built", model=cfg.model, provider=cfg.provider)
    _reranker_resolved = True
    return _reranker


# ── Search result types ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class DocumentContext:
    """L1 document metadata attached to every :class:`SearchHit`."""

    doc_id: str
    title: str
    summary: str


@dataclasses.dataclass(frozen=True)
class SearchHit:
    """One ranked result from :func:`search_knowledge`."""

    topic_id: str
    category_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str
    content: str | None
    score: float
    retrieval_method: str
    source: str | None
    document: DocumentContext


@dataclasses.dataclass(frozen=True)
class SearchKnowledgeResult:
    """Envelope returned by :func:`search_knowledge`."""

    hits: list[SearchHit]
    total: int
    took_ms: float


# ── Where-clause builder ─────────────────────────────────────────────────────


def _validate_scope_id(value: str, name: str) -> None:
    """Reject scope ids with characters that could break LanceDB SQL.

    Args:
        value: The identifier value to validate.
        name: The parameter name (for error messages).

    Raises:
        ValueError: If the value is empty or contains invalid characters.
    """
    if not value or not _SCOPE_ID_PATTERN.match(value):
        raise ValueError(f"{name} contains invalid characters: {value!r}")


def compile_knowledge_where(app_id: str, project_id: str) -> str:
    """Build a LanceDB ``where`` clause scoped to the given tenant.

    Args:
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.

    Returns:
        SQL-style predicate string safe for use in LanceDB ``where`` parameter.

    Raises:
        ValueError: If either id contains invalid characters.
    """
    _validate_scope_id(app_id, "app_id")
    _validate_scope_id(project_id, "project_id")

    def _esc(v: str) -> str:
        return v.replace("'", "''")

    return f"app_id = '{_esc(app_id)}' AND project_id = '{_esc(project_id)}'"


# ── Recall helpers ───────────────────────────────────────────────────────────


async def _base_retrieve(
    recaller: KnowledgeTopicRecaller,
    where: str,
    *,
    method: str,
    query: str,
    vector: list[float],
    limit: int,
) -> list[Candidate]:
    """Run the appropriate recall path and return ranked candidates."""
    if method == "keyword":
        return await recaller.sparse_recall(query, where, limit=limit)
    if method == "vector":
        return await recaller.dense_recall(vector, where, limit=limit)
    # hybrid: parallel sparse + dense, fuse with RRF
    sparse, dense = await asyncio.gather(
        recaller.sparse_recall(query, where, limit=limit),
        recaller.dense_recall(vector, where, limit=limit),
    )
    return rrf(sparse, dense)[:limit]


async def _enrich_with_content(
    candidates: list[Candidate],
) -> list[Candidate]:
    """Batch-fetch SQLite content and attach to candidate metadata.

    Acts as a no-reranker path: returns candidates in their original
    order with ``content`` added to metadata for downstream hit
    conversion.
    """
    if not candidates:
        return candidates

    topic_ids = [c.id for c in candidates]
    topics = await knowledge_topic_sqlite_repo.get_topics_by_ids(topic_ids)
    content_map = {t.node_id: t.content for t in topics}

    return [
        c.model_copy(
            update={
                "metadata": {**c.metadata, "content": content_map.get(c.id, "")},
            }
        )
        for c in candidates
    ]


async def _to_search_hits(
    candidates: list[Candidate],
    include_content: bool,
    method: str,
) -> list[SearchHit]:
    """Convert ranked ``Candidate`` list into ``SearchHit`` DTOs."""
    if not candidates:
        return []

    doc_ids = {c.metadata.get("doc_id", "") for c in candidates} - {""}
    docs = await knowledge_document_repo.get_documents_by_ids(doc_ids)
    doc_map = {d.doc_id: d for d in docs}

    hits: list[SearchHit] = []
    for c in candidates:
        doc = doc_map.get(c.metadata.get("doc_id", ""))
        content = c.metadata.get("content", "") if include_content else None
        source = c.source if c.source != "other" else None
        hits.append(
            SearchHit(
                topic_id=c.id,
                category_id=c.metadata.get("category_id", ""),
                topic_name=c.metadata.get("topic_name", ""),
                topic_path=c.metadata.get("topic_path", ""),
                depth=int(c.metadata.get("depth", 0)),
                summary=c.metadata.get("summary", ""),
                content=content,
                score=c.score,
                retrieval_method=method,
                source=source,
                document=DocumentContext(
                    doc_id=doc.doc_id if doc else "",
                    title=doc.title if doc else "",
                    summary=doc.summary if doc else "",
                ),
            )
        )
    return hits


# ── Public search entry point ────────────────────────────────────────────────


def _require_search_providers() -> tuple[EmbeddingProvider, RerankProvider]:
    """Return embedding + reranker providers, raising if not configured.

    Raises:
        ConfigurationError: When the embedding or rerank provider is not
            configured (a required setting is missing).
    """
    embedder = _get_embedding()
    if embedder is None:
        raise ConfigurationError(
            "Embedding provider not configured. "
            "Set EVEROS_EMBEDDING__MODEL and EVEROS_EMBEDDING__API_KEY."
        )
    reranker = _get_reranker()
    if reranker is None:
        raise ConfigurationError(
            "Rerank provider not configured. "
            "Set EVEROS_RERANK__MODEL and EVEROS_RERANK__BASE_URL."
        )
    return embedder, reranker


async def _run_category_pipeline(
    query: str,
    where: str,
    *,
    method: str,
    vector: list[float],
    reranker: RerankProvider,
    config: KnowledgeSearchSettings,
    top_k: int,
) -> list[Candidate]:
    """Execute the full acategory_retrieve pipeline."""
    from everalgo.rank import acategory_retrieve  # Deferred: heavy dep

    from everos.memory.search.callbacks import (  # Deferred: heavy dep
        build_rerank_fn,
    )

    recaller = _build_recaller()

    async def _retrieve(q: str, k: int) -> list[Candidate]:
        return await _base_retrieve(
            recaller, where, method=method, query=q, vector=vector, limit=k
        )

    raw_rerank = build_rerank_fn(reranker, text_field="content")

    async def _rerank_with_enrich(
        q: str, candidates: Sequence[Candidate]
    ) -> list[Candidate]:
        enriched = await _enrich_with_content(list(candidates))
        return await raw_rerank(q, enriched)

    effective_k = min(top_k, config.top_k_cap)
    return await acategory_retrieve(
        query,
        base_retrieve=_retrieve,
        rerank_fn=_rerank_with_enrich,
        recall_n=config.recall_n,
        rerank_n=config.rerank_n,
        mass_top_m=config.mass_top_m,
        lam=config.lam,
        top_n=effective_k,
    )


async def search_knowledge(
    *,
    query: str,
    method: str = "hybrid",
    top_k: int = 10,
    score_threshold: float | None = None,
    include_content: bool = False,
    app_id: str = "default",
    project_id: str = "default",
) -> SearchKnowledgeResult:
    """Search knowledge topics by keyword, vector, or hybrid retrieval.

    Args:
        query: User search query string.
        method: Retrieval mode — ``"keyword"``, ``"vector"``, or ``"hybrid"``.
        top_k: Maximum hits to return.
        score_threshold: Drop candidates scoring below this value.
        include_content: When ``True``, populate ``SearchHit.content``.
        app_id: Tenant application identifier.
        project_id: Tenant project identifier.

    Returns:
        SearchKnowledgeResult with ranked hits and timing.
    """
    from everos.config import load_settings  # Deferred: singleton

    t0 = time.monotonic()
    config = load_settings().knowledge.search
    where = compile_knowledge_where(app_id, project_id)

    embedder, reranker = _require_search_providers()
    vector = await embedder.embed(query)

    ranked = await _run_category_pipeline(
        query,
        where,
        method=method,
        vector=vector,
        reranker=reranker,
        config=config,
        top_k=top_k,
    )

    if score_threshold is not None:
        ranked = [c for c in ranked if c.score >= score_threshold]
    hits = await _to_search_hits(ranked, include_content, method)

    took_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "knowledge_search_complete",
        method=method,
        query_len=len(query),
        hits=len(hits),
        took_ms=round(took_ms, 1),
    )
    return SearchKnowledgeResult(hits=hits, total=len(hits), took_ms=took_ms)
