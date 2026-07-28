"""Validation paths for knowledge HTTP API routes.

Tests exercise DTO validation, error mapping, and route-level behavior
without external services (no LLM / no LanceDB / no embedder). Service
functions are mocked to isolate the presentation layer.

White-box surfaces: none — all assertions are on HTTP responses.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.infra.persistence.lancedb import lancedb_manager
from everos.service import (
    DocumentDetail,
    DocumentListResult,
    DocumentNotFoundError,
    TopicDetail,
    TopicNotFoundError,
    TopicOverview,
)

knowledge_service_mod = import_module("everos.service.knowledge")

# The route module binds service functions at import time, so patches
# must target the name in the route module's namespace.
_ROUTE_MOD = "everos.entrypoints.api.routes.knowledge"


@pytest.fixture
async def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """FastAPI app with no lifespan; resets knowledge singletons per test.

    The knowledge router gates every route on the embed + rerank
    capability singletons (Task 13). This suite exercises DTO
    validation / route mapping, not the capability gate itself, so
    both singletons are stubbed available here -- otherwise every
    request would short-circuit into the 422 gate response instead
    of reaching the mocked service call under test.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()

    import everos.component.embedding.accessor as embedding_acc
    import everos.component.rerank.accessor as rerank_acc
    from everos.component.embedding import EmbeddingCapability
    from everos.component.rerank import RerankCapability

    monkeypatch.setattr(
        embedding_acc, "_capability", EmbeddingCapability(provider=object())
    )
    monkeypatch.setattr(rerank_acc, "_capability", RerankCapability(provider=object()))

    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    for attr in ("_embedding", "_reranker"):
        setattr(knowledge_service_mod, attr, None)
    for attr in ("_embedding_resolved", "_reranker_resolved"):
        setattr(knowledge_service_mod, attr, False)

    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await lancedb_manager.dispose_connection()
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
        f"{_ROUTE_MOD}.get_document",
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
        f"{_ROUTE_MOD}.get_document",
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
        f"{_ROUTE_MOD}.get_topic",
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
        f"{_ROUTE_MOD}.get_topic",
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
        f"{_ROUTE_MOD}.list_categories",
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
        f"{_ROUTE_MOD}.list_documents",
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
    with patch(f"{_ROUTE_MOD}.list_documents", mock):
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
    with patch(f"{_ROUTE_MOD}.list_documents", mock):
        resp = await client.get(
            "/api/v1/knowledge/documents",
            params={"sort_by": "updated_at"},
        )

    assert resp.status_code == 200
    assert mock.call_args.kwargs["sort_by"] == "updated_at"


def test_reject_oversized_upload() -> None:
    """Uploads above max_upload_bytes are rejected; smaller/unknown pass."""
    from types import SimpleNamespace

    from everos.core.errors import InvalidInputError
    from everos.entrypoints.api.routes.knowledge import _reject_oversized_upload

    over_limit = SimpleNamespace(size=load_settings().knowledge.max_upload_bytes + 1)
    with pytest.raises(InvalidInputError, match="exceeds"):
        _reject_oversized_upload(over_limit)  # type: ignore[arg-type]  # duck-typed stub

    _reject_oversized_upload(SimpleNamespace(size=1024))  # type: ignore[arg-type]
    _reject_oversized_upload(SimpleNamespace(size=None))  # type: ignore[arg-type]


# ── DELETE /documents/{doc_id} ──────────────────────────────────────────────


async def test_delete_document_returns_204_when_not_found(
    client: AsyncClient,
) -> None:
    """Idempotent delete: nonexistent doc returns 204."""
    from everos.service import DeleteResult

    result = DeleteResult(doc_id="d_111111111111", deleted_topics=0)
    with patch(
        f"{_ROUTE_MOD}.delete_document",
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
        f"{_ROUTE_MOD}.delete_document",
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
    from everalgo.types import ParsedContent

    with (
        patch(
            f"{_ROUTE_MOD}._parse_upload",
            new_callable=AsyncMock,
            return_value=ParsedContent(text="# Hello"),
        ),
        patch(
            f"{_ROUTE_MOD}._build_extractor",
            return_value=AsyncMock(),
        ),
        patch(
            f"{_ROUTE_MOD}.replace_document",
            new_callable=AsyncMock,
            side_effect=DocumentNotFoundError("d_000000000000"),
        ),
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
        f"{_ROUTE_MOD}.patch_document",
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
        f"{_ROUTE_MOD}.patch_document",
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
        f"{_ROUTE_MOD}.list_documents",
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


# ── _parse_upload: binary file rejection ────────────────────────────────────


def _hide_everalgo_parser() -> MagicMock:
    """Return a sys.modules patch that makes ``everalgo.parser`` unimportable."""
    fake_modules = {k: v for k, v in sys.modules.items()}
    fake_modules["everalgo.parser"] = None  # type: ignore[assignment]
    return fake_modules


async def test_post_binary_file_without_parser_returns_415(
    client: AsyncClient,
) -> None:
    """Non-UTF-8 binary file without parser → UnsupportedModalityError → 415."""
    # ``parser_available()`` is @lru_cache'd — clear it so this test's
    # sys.modules patch actually reaches the import check instead of
    # hitting a stale cached True/False from an earlier test in the
    # same worker (round-4 review M7).
    from everos.component.parser import parser_available

    parser_available.cache_clear()
    with patch.dict(sys.modules, {"everalgo.parser": None}):  # type: ignore[dict-item]
        resp = await client.post(
            "/api/v1/knowledge/documents",
            files={
                "file": ("test.bin", b"\x80\x81\x82\x83", "application/octet-stream")
            },
            data={"title": "Binary Test"},
        )
    parser_available.cache_clear()  # leave clean for the next test
    assert resp.status_code == 415


async def test_put_binary_file_without_parser_returns_415(
    client: AsyncClient,
) -> None:
    """Non-UTF-8 binary file on PUT → UnsupportedModalityError → 415."""
    from everos.component.parser import parser_available

    parser_available.cache_clear()
    with patch.dict(sys.modules, {"everalgo.parser": None}):  # type: ignore[dict-item]
        resp = await client.put(
            "/api/v1/knowledge/documents/d_aabbccddee01",
            files={
                "file": ("test.bin", b"\x80\x81\x82\x83", "application/octet-stream")
            },
            data={"title": "Binary Test"},
        )
    parser_available.cache_clear()
    assert resp.status_code == 415


# ── _looks_like_utf8_text: plain-text short-circuit ─────────────────────────


class _UploadStub:
    """Minimal UploadFile-shaped stand-in for _looks_like_utf8_text."""

    def __init__(self, mime: str | None, filename: str | None) -> None:
        self.content_type = mime
        self.filename = filename


@pytest.mark.parametrize(
    "mime, filename, expected",
    [
        # Explicit plain-text mimes: short-circuit
        ("text/markdown", "note.md", True),
        ("text/plain", "note.txt", True),
        ("TEXT/MARKDOWN", "note.md", True),  # case-insensitive
        ("text/x-rst", "readme.rst", True),
        # HTML must NOT short-circuit — needs everalgo's clean_html_for_llm
        # (strips <script>/<style>/<nav>/comments) + 1 MiB cap. Was a
        # regression when the whitelist used ``startswith("text/")``.
        ("text/html", "page.html", False),
        ("TEXT/HTML", "page.html", False),
        # Missing / octet-stream mime + known plaintext extension: OK
        ("application/octet-stream", "note.md", True),
        (None, "note.md", True),
        ("", "notes.rst", True),
        # Unknown extension: fall through
        ("application/octet-stream", "note.bin", False),
        (None, "note.bin", False),
        # Non-text mimes always route to parser
        ("application/pdf", "doc.pdf", False),
        ("image/png", "img.png", False),
        ("application/json", "data.json", False),
        ("text/xml", "data.xml", False),  # future text/* additions default to safe
    ],
)
def test_looks_like_utf8_text_matrix(
    mime: str | None, filename: str, expected: bool
) -> None:
    from everos.entrypoints.api.routes.knowledge import _looks_like_utf8_text

    assert _looks_like_utf8_text(_UploadStub(mime, filename)) is expected  # type: ignore[arg-type]


async def test_utf8_short_circuit_skips_parser_when_available(
    client: AsyncClient,
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
    # `parser_available` / `aparse_file` are locally imported inside
    # _parse_upload (deferred), so patch the source modules — patches on
    # the route module namespace won't intercept a `from … import` that
    # happens at call time.
    with (
        patch(
            "everos.component.parser.parser_available",
            return_value=True,
        ),
        patch(
            "everos.component.parser.aparse_file",
            new=AsyncMock(side_effect=AssertionError("parser must not be called")),
            create=True,
        ),
        patch(
            f"{_ROUTE_MOD}._build_extractor",
            return_value=MagicMock(),
        ),
        patch(
            f"{_ROUTE_MOD}.create_document",
            new=AsyncMock(return_value=result),
        ),
    ):
        resp = await client.post(
            "/api/v1/knowledge/documents",
            files={"file": ("note.md", b"# Hello\n\nplain text.", "text/markdown")},
            data={"title": "MD Test"},
        )
    assert resp.status_code == 201
