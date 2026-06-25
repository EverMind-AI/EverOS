"""Unit tests for per-type exception handlers registered via register_handlers().

Tests use a minimal FastAPI app with synthetic routes that raise specific
exceptions. The full handler suite is wired in via ``register_handlers()``.

White-box surfaces: none — assertions are purely on HTTP response shape.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.core.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmbeddingServiceError,
    ExtractionEmptyError,
    FilterError,
    LLMServiceError,
    MultimodalNotEnabledError,
    PathTraversalError,
    StorageError,
    UnsupportedModalityError,
)
from everos.entrypoints.api.exception_handlers import register_handlers

# ---------------------------------------------------------------------------
# Fixture: minimal app with one route per exception type
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with synthetic routes that raise each error."""
    app = FastAPI()
    register_handlers(app)

    @app.get("/raise/not-found")
    async def _not_found() -> None:
        raise DocumentNotFoundError("doc_abc123")

    @app.get("/raise/conflict")
    async def _conflict() -> None:
        raise DuplicateDocumentError("doc_abc123 already exists")

    @app.get("/raise/extraction-empty")
    async def _extraction_empty() -> None:
        raise ExtractionEmptyError("extraction produced no output")

    @app.get("/raise/filter-error")
    async def _filter_error() -> None:
        raise FilterError("unknown field: foo")

    @app.get("/raise/path-traversal")
    async def _path_traversal() -> None:
        raise PathTraversalError("../etc/passwd")

    @app.get("/raise/unsupported-modality")
    async def _unsupported_modality() -> None:
        raise UnsupportedModalityError("video not supported")

    @app.get("/raise/multimodal-not-enabled")
    async def _multimodal_not_enabled() -> None:
        raise MultimodalNotEnabledError("multimodal extra not installed")

    @app.get("/raise/storage")
    async def _storage() -> None:
        raise StorageError("disk full")

    @app.get("/raise/embedding")
    async def _embedding() -> None:
        raise EmbeddingServiceError("embedding provider timeout")

    @app.get("/raise/llm")
    async def _llm() -> None:
        raise LLMServiceError("LLM rate limit exceeded")

    @app.get("/raise/runtime")
    async def _runtime() -> None:
        raise RuntimeError("oops")

    @app.post("/raise/request-validation")
    async def _request_validation(body: dict) -> None:
        # Starlette raises RequestValidationError automatically when
        # a JSON body is expected but not provided.
        pass  # pragma: no cover

    return app


@pytest.fixture
async def client() -> AsyncClient:
    """Async test client against the minimal exception-handler app.

    ``raise_app_exceptions=False`` lets ServerErrorMiddleware send the 500
    response before re-raising, so the test sees the JSON body rather than
    the raw exception.  In production an ASGI server (uvicorn) absorbs the
    re-raise for its own logging; the client already received the response.
    """
    app = _make_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_envelope(data: dict, *, path: str) -> None:
    """Assert the canonical envelope shape is present."""
    assert "request_id" in data, "envelope must have request_id"
    assert len(data["request_id"]) == 32, "request_id must be 32 hex chars"
    error = data.get("error", {})
    assert "code" in error
    assert "message" in error
    assert "timestamp" in error
    assert "path" in error
    assert error["path"] == path


# ---------------------------------------------------------------------------
# Status code + error.code tests
# ---------------------------------------------------------------------------


async def test_not_found_returns_404(client: AsyncClient) -> None:
    """DocumentNotFoundError → HTTP 404, code=NOT_FOUND."""
    resp = await client.get("/raise/not-found")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"]["code"] == "NOT_FOUND"
    _assert_envelope(data, path="/raise/not-found")


async def test_conflict_returns_409(client: AsyncClient) -> None:
    """DuplicateDocumentError → HTTP 409, code=CONFLICT."""
    resp = await client.get("/raise/conflict")
    assert resp.status_code == 409
    data = resp.json()
    assert data["error"]["code"] == "CONFLICT"
    _assert_envelope(data, path="/raise/conflict")


async def test_extraction_empty_returns_422(client: AsyncClient) -> None:
    """ExtractionEmptyError → HTTP 422, code=EXTRACTION_EMPTY."""
    resp = await client.get("/raise/extraction-empty")
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "EXTRACTION_EMPTY"
    _assert_envelope(data, path="/raise/extraction-empty")


async def test_filter_error_returns_422(client: AsyncClient) -> None:
    """FilterError (subclass of InvalidInputError) → HTTP 422."""
    resp = await client.get("/raise/filter-error")
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "INVALID_INPUT"
    _assert_envelope(data, path="/raise/filter-error")


async def test_path_traversal_returns_400(client: AsyncClient) -> None:
    """PathTraversalError → HTTP 400, code=BAD_REQUEST."""
    resp = await client.get("/raise/path-traversal")
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "BAD_REQUEST"
    _assert_envelope(data, path="/raise/path-traversal")


async def test_unsupported_modality_returns_415(client: AsyncClient) -> None:
    """UnsupportedModalityError → HTTP 415, code=UNSUPPORTED_MEDIA_TYPE."""
    resp = await client.get("/raise/unsupported-modality")
    assert resp.status_code == 415
    data = resp.json()
    assert data["error"]["code"] == "UNSUPPORTED_FORMAT"
    _assert_envelope(data, path="/raise/unsupported-modality")


async def test_multimodal_not_enabled_returns_503(client: AsyncClient) -> None:
    """MultimodalNotEnabledError (CapabilityError) → HTTP 503."""
    resp = await client.get("/raise/multimodal-not-enabled")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"]["code"] == "CAPABILITY_UNAVAILABLE"
    _assert_envelope(data, path="/raise/multimodal-not-enabled")


async def test_storage_error_returns_503(client: AsyncClient) -> None:
    """StorageError → HTTP 503, code=SERVICE_UNAVAILABLE."""
    resp = await client.get("/raise/storage")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"]["code"] == "EXTERNAL_SERVICE_UNAVAILABLE"
    _assert_envelope(data, path="/raise/storage")


# ---------------------------------------------------------------------------
# MRO dispatch tests
# ---------------------------------------------------------------------------


async def test_embedding_service_error_routes_to_503(client: AsyncClient) -> None:
    """EmbeddingServiceError (InfrastructureError subclass) → 503 via MRO."""
    resp = await client.get("/raise/embedding")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"]["code"] == "EXTERNAL_SERVICE_UNAVAILABLE"
    _assert_envelope(data, path="/raise/embedding")


async def test_llm_service_error_routes_to_503(client: AsyncClient) -> None:
    """LLMServiceError (InfrastructureError subclass) → 503 via MRO."""
    resp = await client.get("/raise/llm")
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"]["code"] == "EXTERNAL_SERVICE_UNAVAILABLE"
    _assert_envelope(data, path="/raise/llm")


# ---------------------------------------------------------------------------
# Unexpected exception test
# ---------------------------------------------------------------------------


async def test_unexpected_exception_returns_500(client: AsyncClient) -> None:
    """RuntimeError → HTTP 500, INTERNAL_ERROR, generic message (no leak)."""
    resp = await client.get("/raise/runtime")
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"]["code"] == "INTERNAL_ERROR"
    assert data["error"]["message"] == "Internal server error"
    _assert_envelope(data, path="/raise/runtime")


# ---------------------------------------------------------------------------
# RequestValidationError test
# ---------------------------------------------------------------------------


async def test_request_validation_error_returns_422(client: AsyncClient) -> None:
    """RequestValidationError (FastAPI body parse failure) → 422 VALIDATION_ERROR."""
    # POST with invalid JSON triggers RequestValidationError automatically.
    resp = await client.post(
        "/raise/request-validation",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "INVALID_INPUT"
    _assert_envelope(data, path="/raise/request-validation")


# ---------------------------------------------------------------------------
# Envelope shape tests (request_id, timestamp, path)
# ---------------------------------------------------------------------------


async def test_envelope_has_request_id_32_hex(client: AsyncClient) -> None:
    """request_id in envelope must be exactly 32 lowercase hex characters."""
    resp = await client.get("/raise/not-found")
    data = resp.json()
    rid = data["request_id"]
    assert len(rid) == 32
    assert rid == rid.lower()
    assert all(c in "0123456789abcdef" for c in rid)


async def test_envelope_has_iso_timestamp(client: AsyncClient) -> None:
    """error.timestamp must be a non-empty ISO 8601 string."""
    resp = await client.get("/raise/storage")
    data = resp.json()
    ts = data["error"]["timestamp"]
    assert isinstance(ts, str)
    assert len(ts) > 0
    # ISO 8601 basic sanity: contains 'T' separator
    assert "T" in ts


async def test_envelope_path_matches_request(client: AsyncClient) -> None:
    """error.path must match the actual request path."""
    resp = await client.get("/raise/conflict")
    data = resp.json()
    assert data["error"]["path"] == "/raise/conflict"
