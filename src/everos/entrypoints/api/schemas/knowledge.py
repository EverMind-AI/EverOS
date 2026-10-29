"""Request and response models for the ``/knowledge`` endpoints.

Each response model that mirrors a service result builds itself with
``from_result``; timestamps are rendered in the display timezone there.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from everos.core.utils.datetime import to_display_tz

from .ids import AppScopeId, ProjectScopeId

if TYPE_CHECKING:
    from everos.service import (
        CreateDocumentResult,
        DocumentDetail,
        DocumentListResult,
        SearchKnowledgeResult,
        TopicDetail,
    )

# Upper bound on a search query string — guards the embedding call against
# pathologically long input.
MAX_QUERY_LENGTH = 2000


# ── Response DTOs ────────────────────────────────────────────────────────────


class DocumentCreateResponse(BaseModel):
    """Response for POST/PUT /documents."""

    doc_id: str
    category_id: str
    topic_count: int
    source_name: str | None
    md_path: str
    original_file_path: str | None

    @classmethod
    def from_result(cls, result: CreateDocumentResult) -> DocumentCreateResponse:
        return cls(
            doc_id=result.doc_id,
            category_id=result.category_id,
            topic_count=result.topic_count,
            source_name=result.source_name,
            md_path=result.md_path,
            original_file_path=result.original_file_path,
        )


class DocumentDeleteResponse(BaseModel):
    """Response for DELETE /documents/{doc_id}."""

    doc_id: str
    deleted_topics: int


class TopicOverviewDTO(BaseModel):
    """Minimal topic summary inside a document detail."""

    topic_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str


class DocumentDetailResponse(BaseModel):
    """Response for GET /documents/{doc_id}."""

    doc_id: str
    category_id: str
    title: str
    summary: str
    source_name: str | None
    source_type: str | None
    original_file_path: str | None
    topics: list[TopicOverviewDTO]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_result(cls, detail: DocumentDetail) -> DocumentDetailResponse:
        return cls(
            doc_id=detail.doc_id,
            category_id=detail.category_id,
            title=detail.title,
            summary=detail.summary,
            source_name=detail.source_name,
            source_type=detail.source_type,
            original_file_path=detail.original_file_path,
            topics=[
                TopicOverviewDTO(
                    topic_id=t.topic_id,
                    topic_name=t.topic_name,
                    topic_path=t.topic_path,
                    depth=t.depth,
                    summary=t.summary,
                )
                for t in detail.topics
            ],
            created_at=to_display_tz(detail.created_at),
            updated_at=to_display_tz(detail.updated_at),
        )


class DocumentOverviewItemDTO(BaseModel):
    """One row in the paginated document list."""

    doc_id: str
    category_id: str
    title: str
    topic_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""

    documents: list[DocumentOverviewItemDTO]
    total: int
    page: int
    page_size: int

    @classmethod
    def from_result(cls, result: DocumentListResult) -> DocumentListResponse:
        return cls(
            documents=[
                DocumentOverviewItemDTO(
                    doc_id=d.doc_id,
                    category_id=d.category_id,
                    title=d.title,
                    topic_count=d.topic_count,
                    created_at=to_display_tz(d.created_at),
                )
                for d in result.documents
            ],
            total=result.total,
            page=result.page,
            page_size=result.page_size,
        )


class TopicDetailResponse(BaseModel):
    """Response for GET /topics/{topic_id}."""

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

    @classmethod
    def from_result(cls, detail: TopicDetail) -> TopicDetailResponse:
        return cls(
            topic_id=detail.topic_id,
            doc_id=detail.doc_id,
            category_id=detail.category_id,
            topic_name=detail.topic_name,
            topic_path=detail.topic_path,
            depth=detail.depth,
            summary=detail.summary,
            content=detail.content,
            content_labels=detail.content_labels,
            parent_topic_id=detail.parent_topic_id,
            children_topic_ids=detail.children_topic_ids,
            created_at=to_display_tz(detail.created_at),
            updated_at=to_display_tz(detail.updated_at),
        )


class DocumentContextDTO(BaseModel):
    """L1 document metadata attached to every search hit."""

    doc_id: str
    title: str
    summary: str


class SearchHitDTO(BaseModel):
    """One ranked result from knowledge search."""

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
    document: DocumentContextDTO


class KnowledgeSearchResponse(BaseModel):
    """Response for POST /search."""

    hits: list[SearchHitDTO]
    total: int
    took_ms: float

    @classmethod
    def from_result(cls, result: SearchKnowledgeResult) -> KnowledgeSearchResponse:
        return cls(
            hits=[
                SearchHitDTO(
                    topic_id=h.topic_id,
                    category_id=h.category_id,
                    topic_name=h.topic_name,
                    topic_path=h.topic_path,
                    depth=h.depth,
                    summary=h.summary,
                    content=h.content,
                    score=h.score,
                    retrieval_method=h.retrieval_method,
                    source=h.source,
                    document=DocumentContextDTO(
                        doc_id=h.document.doc_id,
                        title=h.document.title,
                        summary=h.document.summary,
                    ),
                )
                for h in result.hits
            ],
            total=result.total,
            took_ms=result.took_ms,
        )


class CategoryDTO(BaseModel):
    """One taxonomy category."""

    category_id: str
    description: str
    document_count: int


class CategoryListResponse(BaseModel):
    """Response for GET /categories."""

    categories: list[CategoryDTO]


class DocumentPatchResponse(BaseModel):
    """Response for PATCH /documents/{doc_id}."""

    doc_id: str
    updated_fields: list[str]
    updated_at: datetime


# ── Request DTOs ─────────────────────────────────────────────────────────────


class KnowledgeSearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    method: Literal["keyword", "vector", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = None
    include_content: bool = False
    app_id: AppScopeId = "default"
    project_id: ProjectScopeId = "default"


class DocumentPatchRequest(BaseModel):
    """Request body for PATCH /documents/{doc_id}."""

    title: str | None = Field(default=None, min_length=1, pattern=r"\w")
    category_id: str | None = Field(default=None, min_length=1)
    app_id: AppScopeId = "default"
    project_id: ProjectScopeId = "default"
