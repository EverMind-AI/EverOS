"""Knowledge document CRUD + search HTTP endpoints.

Routes follow the thin-adapter pattern: validate the DTO, delegate to
the service layer, format the response envelope.  No business logic
lives here.

Endpoint overview (design spec sections 6.2-6.6):
    POST   /documents          — upload a new document (multipart)
    PUT    /documents/{doc_id} — replace an existing document (multipart)
    PATCH  /documents/{doc_id} — update mutable metadata
    DELETE /documents/{doc_id} — remove a document
    GET    /documents          — paginated document listing
    GET    /documents/{doc_id} — single document detail
    GET    /topics/{topic_id}  — single topic detail
    POST   /search             — knowledge retrieval
    GET    /categories         — taxonomy listing
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from everos.service.knowledge import KnowledgeExtractor

from everalgo.types import ParsedContent
from fastapi import APIRouter, Path, Query, Request, Response, UploadFile
from fastapi.params import Form
from pydantic import BaseModel, Field

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import to_display_tz
from everos.config import load_settings
from everos.core.errors import (
    InvalidInputError,
    UnsupportedModalityError,
)
from everos.core.persistence import MemoryRoot
from everos.entrypoints.api.utils import extract_request_id
from everos.service import (
    CreateDocumentResult,
    DocumentDetail,
    DocumentListResult,
    SearchKnowledgeResult,
    TopicDetail,
    create_document,
    delete_document,
    get_document,
    get_topic,
    list_categories,
    list_documents,
    patch_document,
    replace_document,
    search_knowledge,
)

# PathSafeId and SuccessEnvelope are imported from memorize routes;
# a shared module would be cleaner but is out of scope for this PR.
from .memorize import PathSafeId, SuccessEnvelope

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


# ── Annotated param types (satisfies B008) ──────────────────────────────────

_FormTitle = Annotated[str, Form(min_length=1, pattern=r"\w")]
_FormOptStr = Annotated[str | None, Form()]
_FormPathSafe = Annotated[PathSafeId, Form()]
_QueryPathSafe = Annotated[PathSafeId, Query()]
_QueryOptStr = Annotated[str | None, Query()]
_QueryPage = Annotated[int, Query(ge=1)]
_QueryPageSize = Annotated[int, Query(ge=1, le=100)]
_QuerySortBy = Annotated[Literal["created_at", "updated_at", "title"], Query()]
_QuerySortOrder = Annotated[Literal["asc", "desc"], Query()]

# Upper bound on a search query string — guards the embedding call against
# pathologically long input.
_MAX_QUERY_LENGTH = 2000

# doc_id: system-generated "d_<hex12..32>"; topic_id: "{doc_id}_{index}".
_DOC_ID_PATTERN = r"^d_[a-f0-9]{12,32}$"
_TOPIC_ID_PATTERN = r"^d_[a-f0-9]{12,32}_\d+$"
_PathDocId = Annotated[str, Path(pattern=_DOC_ID_PATTERN)]
_PathTopicId = Annotated[str, Path(pattern=_TOPIC_ID_PATTERN)]


# ── Response DTOs ────────────────────────────────────────────────────────────


class DocumentCreateResponse(BaseModel):
    """Response for POST/PUT /documents."""

    doc_id: str
    category_id: str
    topic_count: int
    source_name: str | None
    md_path: str
    original_file_path: str | None


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

    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LENGTH)
    method: Literal["keyword", "vector", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = None
    include_content: bool = False
    app_id: PathSafeId = "default"
    project_id: PathSafeId = "default"


class DocumentPatchRequest(BaseModel):
    """Request body for PATCH /documents/{doc_id}."""

    title: str | None = Field(default=None, min_length=1, pattern=r"\w")
    category_id: str | None = Field(default=None, min_length=1)
    app_id: PathSafeId = "default"
    project_id: PathSafeId = "default"


# ── Extractor builder ───────────────────────────────────────────────────────


def _build_extractor() -> KnowledgeExtractor:
    """Lazily import and build the knowledge extractor from ``everalgo``.

    Returns an object satisfying the ``KnowledgeExtractor`` protocol
    defined in ``service.knowledge``.
    """
    # Deferred: heavy everalgo import; only needed on document creation.
    from everalgo.knowledge import KnowledgeExtractor as AlgoKnowledgeExtractor

    return AlgoKnowledgeExtractor(llm=get_llm_client())


# ── Helpers ──────────────────────────────────────────────────────────────────


def _reject_oversized_upload(file: UploadFile) -> None:
    """Reject an upload whose declared size exceeds the configured limit.

    Raises:
        InvalidInputError: When ``file.size`` exceeds ``knowledge.max_upload_bytes``.
    """
    max_bytes = load_settings().knowledge.max_upload_bytes
    if file.size is not None and file.size > max_bytes:
        limit_mib = max_bytes / (1024 * 1024)
        raise InvalidInputError(f"Uploaded file exceeds the {limit_mib:.1f} MiB limit.")


async def _parse_upload(
    file: UploadFile, *, raw_bytes: bytes | None = None
) -> ParsedContent:
    """Parse an uploaded file via component.parser, or fall back to UTF-8.

    Args:
        file: FastAPI upload file handle.
        raw_bytes: Pre-read bytes; when provided, skips ``file.read()``.

    Raises:
        UnsupportedModalityError: When the file cannot be parsed.
        InvalidInputError: When the parsed content is empty.
    """
    from everos.component.parser import parser_available  # Deferred: optional dep

    if raw_bytes is None:
        raw_bytes = await file.read()

    if parser_available():
        from everalgo.types import RawFile  # Deferred: optional dep

        from everos.component.parser import aparse_file  # Deferred: optional dep

        extension = ""
        if file.filename and "." in file.filename:
            extension = file.filename.rsplit(".", 1)[-1].lower()
        parsed = await aparse_file(
            RawFile(
                content=raw_bytes,
                mime=file.content_type or "",
                extension=extension,
            )
        )
        if not parsed.text or not parsed.text.strip():
            raise InvalidInputError("Uploaded file has no valid content.")
        return parsed

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedModalityError(
            "File is not UTF-8 text. "
            "Install everos[multimodal] for PDF/HTML/DOCX support."
        ) from exc
    parsed = ParsedContent(text=text)
    if not parsed.text or not parsed.text.strip():
        raise InvalidInputError("Uploaded file has no valid content.")
    return parsed


def _map_create_result(result: CreateDocumentResult) -> DocumentCreateResponse:
    """Map CreateDocumentResult to response DTO."""
    return DocumentCreateResponse(
        doc_id=result.doc_id,
        category_id=result.category_id,
        topic_count=result.topic_count,
        source_name=result.source_name,
        md_path=result.md_path,
        original_file_path=result.original_file_path,
    )


def _map_document_detail(detail: DocumentDetail) -> DocumentDetailResponse:
    """Map DocumentDetail to response DTO."""
    return DocumentDetailResponse(
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


def _map_list_result(result: DocumentListResult) -> DocumentListResponse:
    """Map DocumentListResult to response DTO."""
    return DocumentListResponse(
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


def _map_topic_detail(detail: TopicDetail) -> TopicDetailResponse:
    """Map TopicDetail to response DTO."""
    return TopicDetailResponse(
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


def _map_search_result(result: SearchKnowledgeResult) -> KnowledgeSearchResponse:
    """Map SearchKnowledgeResult to response DTO."""
    return KnowledgeSearchResponse(
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


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/documents", status_code=201)
# FastAPI requires flat Form/Query params — ≤5 positional rule exempted.
async def create_document_route(
    request: Request,
    file: UploadFile,
    title: _FormTitle,
    source_type: _FormOptStr = None,
    category_id: _FormOptStr = None,
    app_id: _FormPathSafe = "default",
    project_id: _FormPathSafe = "default",
) -> SuccessEnvelope[DocumentCreateResponse]:
    """Upload a new knowledge document."""
    rid = extract_request_id(request)
    _reject_oversized_upload(file)
    file_content = await file.read()
    parsed = await _parse_upload(file, raw_bytes=file_content)
    source_name = file.filename

    knowledge_dir = MemoryRoot.default().knowledge_dir(app_id, project_id)
    extractor = _build_extractor()

    result = await create_document(
        extractor=extractor,
        parsed=parsed,
        title=title,
        knowledge_dir=knowledge_dir,
        source_name=source_name,
        source_type=source_type,
        category_id=category_id,
        file_content=file_content,
    )
    return SuccessEnvelope(request_id=rid, data=_map_create_result(result))


@router.put("/documents/{doc_id}", status_code=200)
# FastAPI requires flat Form/Query params — ≤5 positional rule exempted.
async def replace_document_route(
    request: Request,
    doc_id: _PathDocId,
    file: UploadFile,
    title: _FormTitle,
    source_type: _FormOptStr = None,
    category_id: _FormOptStr = None,
    app_id: _FormPathSafe = "default",
    project_id: _FormPathSafe = "default",
) -> SuccessEnvelope[DocumentCreateResponse]:
    """Replace an existing knowledge document (atomic backup/restore on failure)."""
    rid = extract_request_id(request)
    _reject_oversized_upload(file)
    file_content = await file.read()
    parsed = await _parse_upload(file, raw_bytes=file_content)

    knowledge_dir = MemoryRoot.default().knowledge_dir(app_id, project_id)
    extractor = _build_extractor()

    result = await replace_document(
        extractor=extractor,
        parsed=parsed,
        title=title,
        doc_id=doc_id,
        knowledge_dir=knowledge_dir,
        source_name=file.filename,
        source_type=source_type,
        category_id=category_id,
        file_content=file_content,
    )
    return SuccessEnvelope(request_id=rid, data=_map_create_result(result))


@router.delete("/documents/{doc_id}", response_model=None)
async def delete_document_route(
    request: Request,
    doc_id: _PathDocId,
    app_id: _QueryPathSafe = "default",
    project_id: _QueryPathSafe = "default",
) -> SuccessEnvelope[DocumentDeleteResponse] | Response:
    """Remove a knowledge document."""
    rid = extract_request_id(request)
    result = await delete_document(doc_id, app_id, project_id)

    if result.deleted_topics == 0:
        return Response(status_code=204)

    return SuccessEnvelope(
        request_id=rid,
        data=DocumentDeleteResponse(
            doc_id=result.doc_id,
            deleted_topics=result.deleted_topics,
        ),
    )


@router.get("/documents")
# FastAPI requires flat Form/Query params — ≤5 positional rule exempted.
async def list_documents_route(
    request: Request,
    app_id: _QueryPathSafe = "default",
    project_id: _QueryPathSafe = "default",
    category_id: _QueryOptStr = None,
    page: _QueryPage = 1,
    page_size: _QueryPageSize = 20,
    sort_by: _QuerySortBy = "created_at",
    sort_order: _QuerySortOrder = "desc",
) -> SuccessEnvelope[DocumentListResponse]:
    """Paginated document listing."""
    rid = extract_request_id(request)
    result = await list_documents(
        app_id,
        project_id,
        category_id=category_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return SuccessEnvelope(request_id=rid, data=_map_list_result(result))


@router.get("/documents/{doc_id}")
async def get_document_route(
    request: Request,
    doc_id: _PathDocId,
    app_id: _QueryPathSafe = "default",
    project_id: _QueryPathSafe = "default",
) -> SuccessEnvelope[DocumentDetailResponse]:
    """Fetch a single document with its topic list."""
    rid = extract_request_id(request)
    detail = await get_document(doc_id, app_id, project_id)
    return SuccessEnvelope(request_id=rid, data=_map_document_detail(detail))


@router.get("/topics/{topic_id}")
async def get_topic_route(
    request: Request,
    topic_id: _PathTopicId,
    app_id: _QueryPathSafe = "default",
    project_id: _QueryPathSafe = "default",
) -> SuccessEnvelope[TopicDetailResponse]:
    """Fetch a single topic with full content."""
    rid = extract_request_id(request)
    detail = await get_topic(topic_id, app_id, project_id)
    return SuccessEnvelope(request_id=rid, data=_map_topic_detail(detail))


@router.post("/search")
async def search_knowledge_route(
    request: Request,
    req: KnowledgeSearchRequest,
) -> SuccessEnvelope[KnowledgeSearchResponse]:
    """Knowledge retrieval (keyword / vector / hybrid)."""
    rid = extract_request_id(request)
    result = await search_knowledge(
        query=req.query,
        method=req.method,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        include_content=req.include_content,
        app_id=req.app_id,
        project_id=req.project_id,
    )
    return SuccessEnvelope(request_id=rid, data=_map_search_result(result))


@router.get("/categories")
async def list_categories_route(
    request: Request,
    app_id: _QueryPathSafe = "default",
    project_id: _QueryPathSafe = "default",
) -> SuccessEnvelope[CategoryListResponse]:
    """List taxonomy categories from ``.taxonomy.md``."""
    rid = extract_request_id(request)
    overviews = await list_categories(app_id, project_id)
    categories = [
        CategoryDTO(
            category_id=c.category_id,
            description=c.description,
            document_count=c.document_count,
        )
        for c in overviews
    ]
    return SuccessEnvelope(
        request_id=rid,
        data=CategoryListResponse(categories=categories),
    )


@router.patch("/documents/{doc_id}")
async def patch_document_route(
    request: Request,
    doc_id: _PathDocId,
    req: DocumentPatchRequest,
) -> SuccessEnvelope[DocumentPatchResponse]:
    """Update mutable document metadata fields."""
    rid = extract_request_id(request)
    result = await patch_document(
        doc_id,
        req.app_id,
        req.project_id,
        title=req.title,
        category_id=req.category_id,
    )
    return SuccessEnvelope(
        request_id=rid,
        data=DocumentPatchResponse(
            doc_id=result.doc_id,
            updated_fields=result.updated_fields,
            updated_at=to_display_tz(result.updated_at),
        ),
    )
