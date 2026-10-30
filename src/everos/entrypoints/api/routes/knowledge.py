"""Knowledge document CRUD + search HTTP endpoints.

Routes follow the thin-adapter pattern: validate the DTO, delegate to
the service layer, format the response envelope. Request and response
models live in ``schemas.knowledge``. No business logic lives here.

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

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query, Response, UploadFile
from fastapi.params import Form

from everos.core.utils.datetime import to_display_tz
from everos.entrypoints.api.dependencies import ApplicationDep, RequestIdDep
from everos.entrypoints.api.schemas import (
    AppScopeId,
    CategoryDTO,
    CategoryListResponse,
    DocumentCreateResponse,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentPatchRequest,
    DocumentPatchResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    ProjectScopeId,
    SuccessEnvelope,
    TopicDetailResponse,
)


def _require_knowledge_capabilities(app: ApplicationDep) -> None:
    """Per-endpoint gate: uploads and search need embedding + rerank.

    The rule is :meth:`KnowledgeService.require_search_providers`; it runs as
    a dependency so FastAPI resolves it before validating the body, and a
    deployment without the providers gets the provider-specific 422 for any
    request. Attached per endpoint, not router-wide, so reads, deletes and
    metadata patches stay reachable after a provider is removed.
    """
    app.knowledge.require_search_providers()


# Router prefix is /knowledge; app.py mounts it under both /api/v1 and
# /api/v2 (see create_app — v1 retained as a permanent alias).
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# ── Annotated param types (satisfies B008) ──────────────────────────────────

_FormTitle = Annotated[str, Form(min_length=1, pattern=r"\w")]
_FormOptStr = Annotated[str | None, Form()]
_FormAppId = Annotated[AppScopeId, Form()]
_FormProjectId = Annotated[ProjectScopeId, Form()]
_QueryAppId = Annotated[AppScopeId, Query()]
_QueryProjectId = Annotated[ProjectScopeId, Query()]
_QueryOptStr = Annotated[str | None, Query()]
_QueryPage = Annotated[int, Query(ge=1)]
_QueryPageSize = Annotated[int, Query(ge=1, le=100)]
_QuerySortBy = Annotated[Literal["created_at", "updated_at", "title"], Query()]
_QuerySortOrder = Annotated[Literal["asc", "desc"], Query()]

# doc_id: system-generated "d_<hex12..32>"; topic_id: "{doc_id}_{index}".
_DOC_ID_PATTERN = r"^d_[a-f0-9]{12,32}$"
_TOPIC_ID_PATTERN = r"^d_[a-f0-9]{12,32}_\d+$"
_PathDocId = Annotated[str, Path(pattern=_DOC_ID_PATTERN)]
_PathTopicId = Annotated[str, Path(pattern=_TOPIC_ID_PATTERN)]


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post(
    "/documents",
    status_code=201,
    dependencies=[Depends(_require_knowledge_capabilities)],
)
# FastAPI requires flat Form/Query params — ≤5 positional rule exempted.
async def create_document_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    file: UploadFile,
    title: _FormTitle,
    source_type: _FormOptStr = None,
    category_id: _FormOptStr = None,
    app_id: _FormAppId = "default",
    project_id: _FormProjectId = "default",
) -> SuccessEnvelope[DocumentCreateResponse]:
    """Upload a new knowledge document."""
    result = await app.knowledge.create_from_upload(
        file,
        title=title,
        app_id=app_id,
        project_id=project_id,
        source_type=source_type,
        category_id=category_id,
    )
    return SuccessEnvelope(
        request_id=request_id, data=DocumentCreateResponse.from_result(result)
    )


@router.put(
    "/documents/{doc_id}",
    status_code=200,
    dependencies=[Depends(_require_knowledge_capabilities)],
)
# FastAPI requires flat Form/Query params — ≤5 positional rule exempted.
async def replace_document_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    doc_id: _PathDocId,
    file: UploadFile,
    title: _FormTitle,
    source_type: _FormOptStr = None,
    category_id: _FormOptStr = None,
    app_id: _FormAppId = "default",
    project_id: _FormProjectId = "default",
) -> SuccessEnvelope[DocumentCreateResponse]:
    """Replace an existing knowledge document (atomic backup/restore on failure)."""
    result = await app.knowledge.replace_from_upload(
        doc_id,
        file,
        title=title,
        app_id=app_id,
        project_id=project_id,
        source_type=source_type,
        category_id=category_id,
    )
    return SuccessEnvelope(
        request_id=request_id, data=DocumentCreateResponse.from_result(result)
    )


@router.delete("/documents/{doc_id}", response_model=None)
async def delete_document_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    doc_id: _PathDocId,
    app_id: _QueryAppId = "default",
    project_id: _QueryProjectId = "default",
) -> SuccessEnvelope[DocumentDeleteResponse] | Response:
    """Remove a knowledge document."""
    result = await app.knowledge.delete_document(doc_id, app_id, project_id)

    if result.deleted_topics == 0:
        return Response(status_code=204)

    return SuccessEnvelope(
        request_id=request_id,
        data=DocumentDeleteResponse(
            doc_id=result.doc_id,
            deleted_topics=result.deleted_topics,
        ),
    )


@router.get("/documents")
# FastAPI requires flat Form/Query params — ≤5 positional rule exempted.
async def list_documents_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    app_id: _QueryAppId = "default",
    project_id: _QueryProjectId = "default",
    category_id: _QueryOptStr = None,
    page: _QueryPage = 1,
    page_size: _QueryPageSize = 20,
    sort_by: _QuerySortBy = "created_at",
    sort_order: _QuerySortOrder = "desc",
) -> SuccessEnvelope[DocumentListResponse]:
    """Paginated document listing."""
    result = await app.knowledge.list_documents(
        app_id,
        project_id,
        category_id=category_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return SuccessEnvelope(
        request_id=request_id, data=DocumentListResponse.from_result(result)
    )


@router.get("/documents/{doc_id}")
async def get_document_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    doc_id: _PathDocId,
    app_id: _QueryAppId = "default",
    project_id: _QueryProjectId = "default",
) -> SuccessEnvelope[DocumentDetailResponse]:
    """Fetch a single document with its topic list."""
    detail = await app.knowledge.get_document(doc_id, app_id, project_id)
    return SuccessEnvelope(
        request_id=request_id, data=DocumentDetailResponse.from_result(detail)
    )


@router.get("/topics/{topic_id}")
async def get_topic_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    topic_id: _PathTopicId,
    app_id: _QueryAppId = "default",
    project_id: _QueryProjectId = "default",
) -> SuccessEnvelope[TopicDetailResponse]:
    """Fetch a single topic with full content."""
    detail = await app.knowledge.get_topic(topic_id, app_id, project_id)
    return SuccessEnvelope(
        request_id=request_id, data=TopicDetailResponse.from_result(detail)
    )


@router.post(
    "/search",
    dependencies=[Depends(_require_knowledge_capabilities)],
)
async def search_knowledge_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    req: KnowledgeSearchRequest,
) -> SuccessEnvelope[KnowledgeSearchResponse]:
    """Knowledge retrieval (keyword / vector / hybrid)."""
    result = await app.knowledge.search_knowledge(
        query=req.query,
        method=req.method,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        include_content=req.include_content,
        app_id=req.app_id,
        project_id=req.project_id,
    )
    return SuccessEnvelope(
        request_id=request_id, data=KnowledgeSearchResponse.from_result(result)
    )


@router.get("/categories")
async def list_categories_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    app_id: _QueryAppId = "default",
    project_id: _QueryProjectId = "default",
) -> SuccessEnvelope[CategoryListResponse]:
    """List taxonomy categories from ``.taxonomy.md``."""
    overviews = await app.knowledge.list_categories(app_id, project_id)
    categories = [
        CategoryDTO(
            category_id=c.category_id,
            description=c.description,
            document_count=c.document_count,
        )
        for c in overviews
    ]
    return SuccessEnvelope(
        request_id=request_id,
        data=CategoryListResponse(categories=categories),
    )


@router.patch("/documents/{doc_id}")
async def patch_document_route(
    app: ApplicationDep,
    request_id: RequestIdDep,
    doc_id: _PathDocId,
    req: DocumentPatchRequest,
) -> SuccessEnvelope[DocumentPatchResponse]:
    """Update mutable document metadata fields."""
    result = await app.knowledge.patch_document(
        doc_id,
        req.app_id,
        req.project_id,
        title=req.title,
        category_id=req.category_id,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=DocumentPatchResponse(
            doc_id=result.doc_id,
            updated_fields=result.updated_fields,
            updated_at=to_display_tz(result.updated_at),
        ),
    )
