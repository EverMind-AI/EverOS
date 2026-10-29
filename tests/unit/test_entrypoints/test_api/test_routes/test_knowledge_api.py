"""Validation paths for knowledge HTTP API routes.

Tests exercise DTO validation, error mapping, and route-level behavior
without external services (no LLM / no LanceDB / no embedder). Service
functions are mocked to isolate the presentation layer.

White-box surfaces: none — all assertions are on HTTP responses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import ParsedContent
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.infra.persistence import MemoryRoot
from everos.service import (
    DocumentDetail,
    DocumentListResult,
    DocumentNotFoundError,
    TopicDetail,
    TopicNotFoundError,
    TopicOverview,
)
from tests.fakes import make_application, make_knowledge_service
from tests.fakes.storage import Storage

# Use-case calls are patched on the facade class, which the application's
# KnowledgeService instance resolves.
_SERVICE = "everos.service.knowledge.KnowledgeService"
_CONFIGURED = object()
"""Stands in for a configured embedding / rerank provider (only presence counts)."""


async def _parser_must_not_run(raw_file: object) -> ParsedContent:
    raise AssertionError("parser must not be called")


async def _client_over(
    tmp_path: Path, storage: Storage, *, parser_available: bool
) -> AsyncIterator[AsyncClient]:
    knowledge = make_knowledge_service(
        root=MemoryRoot(tmp_path),
        embedding=_CONFIGURED,
        reranker=_CONFIGURED,
        parse_file=_parser_must_not_run,
        parser_available=parser_available,
        documents=storage.sqlite.knowledge_document,
        topics=storage.sqlite.knowledge_topic,
    )
    app = create_app(application=make_application(knowledge=knowledge))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[AsyncClient]:
    """FastAPI app over a stub application whose knowledge facade is real.

    The knowledge router gates every write / search route on the embed +
    rerank providers. This suite exercises DTO validation / route mapping,
    not the gate itself, so both providers are configured -- otherwise every
    request would short-circuit into the 422 gate response instead of
    reaching the mocked service call under test. The optional parser is
    absent, so non-text uploads are refused.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    async for c in _client_over(tmp_path, storage, parser_available=False):
        yield c
    await storage.lance.close()
    load_settings.cache_clear()


@pytest.fixture
async def client_with_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[AsyncClient]:
    """Same as ``client``, with the optional parser installed (it must not run)."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    async for c in _client_over(tmp_path, storage, parser_available=True):
        yield c
    await storage.lance.close()
    load_settings.cache_clear()


# ── Fixtures ─────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_document_detail(doc_id: str = "d_aabbccddee01123abc123") -> DocumentDetail:
    return DocumentDetail(
        doc_id=doc_id,
        category_id="Technology",
        title="Test Doc",
        summary="A test document.",
        source_name="test.txt",
        source_type="file",
        original_file_path=None,
        topics=[
            TopicOverview(
                topic_id="d_abc123abc123_1",
                topic_name="Intro",
                topic_path="Intro",
                depth=1,
                summary="Introduction section.",
            ),
        ],
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_topic_detail(topic_id: str = "d_abc123abc123_1") -> TopicDetail:
    return TopicDetail(
        topic_id=topic_id,
        doc_id="d_aabbccddee01123abc123",
        category_id="Technology",
        topic_name="Intro",
        topic_path="Intro",
        depth=1,
        summary="Introduction section.",
        content="Some content here.",
        content_labels=["intro"],
        parent_topic_id=None,
        children_topic_ids=["d_abc123abc123_2"],
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── GET /documents/{doc_id} ─────────────────────────────────────────────────


async def test_get_document_success(client: AsyncClient) -> None:
    """Mocked service returns detail; route maps to 200 envelope."""
    detail = _make_document_detail()
    with patch(
        f"{_SERVICE}.get_document",
        new_callable=AsyncMock,
        return_value=detail,
    ):
        resp = await client.get("/api/v1/knowledge/documents/d_aabbccddee01123abc123")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["doc_id"] == "d_aabbccddee01123abc123"
    assert body["data"]["topics"][0]["topic_id"] == "d_abc123abc123_1"
    assert "request_id" in body


async def test_get_document_404(client: AsyncClient) -> None:
    """Nonexistent doc_id returns 404."""
    with patch(
        f"{_SERVICE}.get_document",
        new_callable=AsyncMock,
        side_effect=DocumentNotFoundError("d_000000000000"),
    ):
        resp = await client.get("/api/v1/knowledge/documents/d_000000000000")

    assert resp.status_code == 404


# ── GET /topics/{topic_id} ──────────────────────────────────────────────────


async def test_get_topic_success(client: AsyncClient) -> None:
    """Mocked service returns topic detail; route maps to 200 envelope."""
    detail = _make_topic_detail()
    with patch(
        f"{_SERVICE}.get_topic",
        new_callable=AsyncMock,
        return_value=detail,
    ):
        resp = await client.get("/api/v1/knowledge/topics/d_abc123abc123_1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["topic_id"] == "d_abc123abc123_1"
    assert body["data"]["children_topic_ids"] == ["d_abc123abc123_2"]


async def test_get_topic_404(client: AsyncClient) -> None:
    """Nonexistent topic_id returns 404."""
    with patch(
        f"{_SERVICE}.get_topic",
        new_callable=AsyncMock,
        side_effect=TopicNotFoundError("d_000000000000_999"),
    ):
        resp = await client.get("/api/v1/knowledge/topics/d_000000000000_999")

    assert resp.status_code == 404


# ── POST /search ─────────────────────────────────────────────────────────────


async def test_search_422_empty_query(client: AsyncClient) -> None:
    """Empty query string violates min_length=1."""
    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": ""},
    )
    assert resp.status_code == 422


async def test_search_422_invalid_method(client: AsyncClient) -> None:
    """Invalid method value returns 422."""
    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "hello", "method": "bm42"},
    )
    assert resp.status_code == 422


async def test_search_422_top_k_out_of_range(client: AsyncClient) -> None:
    """top_k=0 or >100 returns 422."""
    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "hello", "top_k": 0},
    )
    assert resp.status_code == 422

    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "hello", "top_k": 101},
    )
    assert resp.status_code == 422


async def test_search_422_query_too_long(client: AsyncClient) -> None:
    """A query beyond max_length is rejected before the embedding call."""
    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "x" * 2001},
    )
    assert resp.status_code == 422


# ── GET /categories ──────────────────────────────────────────────────────────


async def test_get_categories_returns_taxonomy(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    """Returns category list with document counts."""
    from everos.service import CategoryOverview

    overviews = [
        CategoryOverview(
            category_id="Tech", description="Technology topics", document_count=3
        ),
        CategoryOverview(
            category_id="Science", description="Science topics", document_count=0
        ),
    ]
    with patch(
        f"{_SERVICE}.list_categories",
        new=AsyncMock(return_value=overviews),
    ):
        resp = await client.get("/api/v1/knowledge/categories")

    assert resp.status_code == 200
    body = resp.json()
    cats = body["data"]["categories"]
    assert len(cats) == 2
    assert cats[0]["category_id"] == "Tech"
    assert cats[0]["document_count"] == 3
    assert cats[1]["description"] == "Science topics"
    assert cats[1]["document_count"] == 0


# ── GET /documents (list) ───────────────────────────────────────────────────


async def test_list_documents_empty(client: AsyncClient) -> None:
    """Empty result returns paginated envelope with zero items."""
    result = DocumentListResult(documents=[], total=0, page=1, page_size=20)
    with patch(
        f"{_SERVICE}.list_documents",
        new_callable=AsyncMock,
        return_value=result,
    ):
        resp = await client.get("/api/v1/knowledge/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["documents"] == []
    assert body["data"]["total"] == 0


async def test_list_documents_pagination_params(client: AsyncClient) -> None:
    """Pagination params are forwarded to the service."""
    result = DocumentListResult(documents=[], total=0, page=2, page_size=5)
    mock = AsyncMock(return_value=result)
    with patch(f"{_SERVICE}.list_documents", mock):
        resp = await client.get(
            "/api/v1/knowledge/documents",
            params={"page": 2, "page_size": 5, "sort_by": "title", "sort_order": "asc"},
        )

    assert resp.status_code == 200
    mock.assert_called_once_with(
        "default",
        "default",
        category_id=None,
        page=2,
        page_size=5,
        sort_by="title",
        sort_order="asc",
    )


async def test_list_documents_invalid_page_size(client: AsyncClient) -> None:
    """page_size > 100 returns 422."""
    resp = await client.get(
        "/api/v1/knowledge/documents",
        params={"page_size": 200},
    )
    assert resp.status_code == 422


async def test_list_documents_sort_by_updated_at(client: AsyncClient) -> None:
    """sort_by=updated_at is accepted and forwarded (repo supports it)."""
    result = DocumentListResult(documents=[], total=0, page=1, page_size=20)
    mock = AsyncMock(return_value=result)
    with patch(f"{_SERVICE}.list_documents", mock):
        resp = await client.get(
            "/api/v1/knowledge/documents",
            params={"sort_by": "updated_at"},
        )

    assert resp.status_code == 200
    assert mock.call_args.kwargs["sort_by"] == "updated_at"


# ── DELETE /documents/{doc_id} ──────────────────────────────────────────────


async def test_delete_document_returns_204_when_not_found(
    client: AsyncClient,
) -> None:
    """Idempotent delete: nonexistent doc returns 204."""
    from everos.service import DeleteResult

    result = DeleteResult(doc_id="d_111111111111", deleted_topics=0)
    with patch(
        f"{_SERVICE}.delete_document",
        new_callable=AsyncMock,
        return_value=result,
    ):
        resp = await client.delete("/api/v1/knowledge/documents/d_111111111111")

    assert resp.status_code == 204


async def test_delete_document_returns_envelope(client: AsyncClient) -> None:
    """Successful delete with topics returns envelope."""
    from everos.service import DeleteResult

    result = DeleteResult(doc_id="d_aabbccddee01", deleted_topics=3)
    with patch(
        f"{_SERVICE}.delete_document",
        new_callable=AsyncMock,
        return_value=result,
    ):
        resp = await client.delete("/api/v1/knowledge/documents/d_aabbccddee01")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["deleted_topics"] == 3


# ── PUT /documents/{doc_id} ───────────────────────────────────────────────


async def test_put_document_404_when_not_found(client: AsyncClient) -> None:
    """PUT on nonexistent doc_id returns 404 (strict replace, not upsert)."""
    with patch(
        f"{_SERVICE}.replace_document",
        new_callable=AsyncMock,
        side_effect=DocumentNotFoundError("d_000000000000"),
    ):
        resp = await client.put(
            "/api/v1/knowledge/documents/d_000000000000",
            files={"file": ("test.md", b"# Hello", "text/markdown")},
            data={"title": "Test"},
        )

    assert resp.status_code == 404


# ── PATCH /documents/{doc_id} ──────────────────────────────────────────────


async def test_patch_document_success(client: AsyncClient) -> None:
    """Successful patch returns updated fields."""
    from everos.service import PatchResult

    result = PatchResult(
        doc_id="d_aabbccddee01", updated_fields=["title"], updated_at=_NOW
    )
    with patch(
        f"{_SERVICE}.patch_document",
        new_callable=AsyncMock,
        return_value=result,
    ):
        resp = await client.patch(
            "/api/v1/knowledge/documents/d_aabbccddee01",
            json={"title": "New Title"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["updated_fields"] == ["title"]


async def test_patch_document_404(client: AsyncClient) -> None:
    """Patching nonexistent doc returns 404."""
    with patch(
        f"{_SERVICE}.patch_document",
        new_callable=AsyncMock,
        side_effect=DocumentNotFoundError("d_000000000000"),
    ):
        resp = await client.patch(
            "/api/v1/knowledge/documents/d_000000000000",
            json={"title": "New Title"},
        )

    assert resp.status_code == 404


# ── PathSafeId validation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/knowledge/documents/bad_format",
        "/api/v1/knowledge/documents/'; DROP TABLE--",
        "/api/v1/knowledge/documents/d_ZZZZ00000000",
    ],
)
async def test_invalid_doc_id_format_returns_422(
    client: AsyncClient,
    path: str,
) -> None:
    """doc_id path param must match ``d_[a-f0-9]{12,32}``."""
    resp = await client.get(path)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/knowledge/topics/bad_format",
        "/api/v1/knowledge/topics/d_abc123abc123",
        "/api/v1/knowledge/topics/not_valid_at_all",
    ],
)
async def test_invalid_topic_id_format_returns_422(
    client: AsyncClient,
    path: str,
) -> None:
    """topic_id path param must match ``d_[a-f0-9]{12,32}_\\d+``."""
    resp = await client.get(path)
    assert resp.status_code == 422


async def test_pathsafe_rejects_traversal_in_query(client: AsyncClient) -> None:
    """app_id with '..' in query param is rejected."""
    result = DocumentListResult(documents=[], total=0, page=1, page_size=20)
    with patch(
        f"{_SERVICE}.list_documents",
        new_callable=AsyncMock,
        return_value=result,
    ):
        resp = await client.get(
            "/api/v1/knowledge/documents",
            params={"app_id": ".."},
        )

    assert resp.status_code == 422


async def test_pathsafe_rejects_traversal_in_body(client: AsyncClient) -> None:
    """app_id with '..' in JSON body is rejected."""
    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "hello", "app_id": ".."},
    )
    assert resp.status_code == 422


# ── Upload parsing: binary file rejection ───────────────────────────────────


async def test_post_binary_file_without_parser_returns_415(
    client: AsyncClient,
) -> None:
    """Non-UTF-8 binary file without parser → UnsupportedModalityError → 415."""
    resp = await client.post(
        "/api/v1/knowledge/documents",
        files={"file": ("test.bin", b"\x80\x81\x82\x83", "application/octet-stream")},
        data={"title": "Binary Test"},
    )
    assert resp.status_code == 415


async def test_put_binary_file_without_parser_returns_415(
    client: AsyncClient,
) -> None:
    """Non-UTF-8 binary file on PUT → UnsupportedModalityError → 415."""
    resp = await client.put(
        "/api/v1/knowledge/documents/d_aabbccddee01",
        files={"file": ("test.bin", b"\x80\x81\x82\x83", "application/octet-stream")},
        data={"title": "Binary Test"},
    )
    assert resp.status_code == 415


async def test_utf8_short_circuit_skips_parser_when_available(
    client_with_parser: AsyncClient,
) -> None:
    """Markdown upload must NOT call the parser even when parser_available.

    Regression cover: prior _parse_upload went straight into aparse_file
    whenever ``parser_available()`` returned True, forcing markdown
    uploads to require a [multimodal] LLM config. The short-circuit
    keeps text/* + known-extension uploads on the UTF-8 fast path.
    """
    from everos.service import CreateDocumentResult

    result = CreateDocumentResult(
        doc_id="d_aabbccddee01",
        category_id="Technology",
        topic_count=1,
        source_name="note.md",
        md_path="/tmp/note.md",
    )
    # The parser is installed but refuses to run (see `_client_over`).
    with patch(f"{_SERVICE}.create_document", new=AsyncMock(return_value=result)):
        resp = await client_with_parser.post(
            "/api/v1/knowledge/documents",
            files={"file": ("note.md", b"# Hello\n\nplain text.", "text/markdown")},
            data={"title": "MD Test"},
        )
    assert resp.status_code == 201
