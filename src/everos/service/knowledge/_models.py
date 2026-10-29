"""Knowledge result types, the extractor contract and the search scope filter."""

from __future__ import annotations

__all__ = [
    "CategoryOverview",
    "CreateDocumentResult",
    "DeleteResult",
    "DocumentContext",
    "DocumentDetail",
    "DocumentListResult",
    "DocumentOverviewItem",
    "KnowledgeExtractor",
    "PatchResult",
    "SearchHit",
    "SearchKnowledgeResult",
    "TopicDetail",
    "TopicOverview",
    "compile_knowledge_where",
]

import dataclasses
import re
from datetime import datetime
from typing import Protocol

from everalgo.types import CategorySpec, KnowledgeMemory, ParsedContent

from everos.memory.ports import Filter, all_of, eq

_SCOPE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-@+]+$")


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


# ── Search filter ────────────────────────────────────────────────────────────


def _validate_scope_id(value: str, name: str) -> None:
    """Reject scope ids with characters that could break an index filter.

    Raises:
        ValueError: If the value is empty or contains invalid characters.
    """
    if not value or not _SCOPE_ID_PATTERN.match(value):
        raise ValueError(f"{name} contains invalid characters: {value!r}")


def compile_knowledge_where(app_id: str, project_id: str) -> Filter:
    """Build a backend-neutral filter scoped to the given tenant.

    Raises:
        ValueError: If either id contains invalid characters.
    """
    _validate_scope_id(app_id, "app_id")
    _validate_scope_id(project_id, "project_id")
    return all_of(eq("app_id", app_id), eq("project_id", project_id))
