"""Verify per-endpoint scoping of the knowledge capability gate.

Task 11 (``cascade.registry.build_handlers``) gates the knowledge
cascade handlers off as an atomic pair when embed + rerank aren't
both available — a doc queued for a gated-off kind gets marked
permanently failed by the worker instead of ever reaching LanceDB.
Before this router gate existed, the HTTP layer had no idea any of
that happened: a Tier 1 user's upload returned 200 (the md write
always succeeds) while the document silently never became
searchable.

An earlier iteration attached the gate router-wide, which regressed
downgraded users: a client who dropped from Tier 3 → Tier 2/1 could
no longer even GET or DELETE their own docs, because the router-wide
dependency fired on every request. This module pins the corrected
scoping — writes and search stay gated (they can't succeed end-to-
end without the providers) while reads/deletes stay reachable so
users can inspect and clean up state they already have on disk.

Uses ``httpx.AsyncClient`` + ``ASGITransport`` (not
``fastapi.testclient.TestClient``) to match the project's established
exception-handler test convention (see ``tests/integration/test_api/
test_provider_error_mapping.py``). The knowledge router is mounted
standalone (no lifespan / full app) since the gate fires before any
body is even parsed — sub-dependency resolution runs ahead of
body/query/path validation in FastAPI's request handler, so a bare
request with no payload is enough to reach the gate on every route,
including the multipart upload endpoints.

For endpoints without a gate we stub the service call so the test
proves the request reached the handler (i.e. no 422 from the
missing-provider check), independent of MemoryRoot / SQLite setup.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import everos.component.embedding.accessor as embedding_accessor
import everos.component.rerank.accessor as rerank_accessor
from everos.component.embedding import EmbeddingCapability
from everos.component.rerank import RerankCapability
from everos.entrypoints.api.exception_handlers import register_handlers
from everos.entrypoints.api.routes import knowledge as knowledge_routes

# Endpoints that MUST return 422 when embed or rerank is missing.
# These paths either write new md that cascade will need to embed
# (POST/PUT) or run a search that requires both providers
# (POST /search). PATCH is deliberately NOT here — see the ungated
# list below.
_GATED_ENDPOINTS: list[tuple[str, str]] = [
    ("POST", "/api/v1/knowledge/documents"),
    ("PUT", "/api/v1/knowledge/documents/d_abcdef123456"),
    ("POST", "/api/v1/knowledge/search"),
]

# Endpoints that MUST stay reachable even when both capabilities are
# missing. All of these touch only md + SQLite state that already
# exists on disk — none of them need the embedding or rerank provider.
# PATCH is here because ``patch_document`` only rewrites md frontmatter
# and moves the doc directory when category changes; no embed or rerank
# code runs on that path. A user who downgrades from Tier 3 → Tier 1/2
# must still be able to rename or recategorize the documents they
# created while providers were configured.
_UNGATED_ENDPOINTS: list[tuple[str, str]] = [
    ("DELETE", "/api/v1/knowledge/documents/d_abcdef123456"),
    ("GET", "/api/v1/knowledge/documents"),
    ("GET", "/api/v1/knowledge/documents/d_abcdef123456"),
    ("GET", "/api/v1/knowledge/topics/d_abcdef123456_0"),
    ("GET", "/api/v1/knowledge/categories"),
]


def _build_app() -> FastAPI:
    app = FastAPI()
    register_handlers(app)
    # Match production mounting: create_app() mounts the router under
    # /api/v1 (and /api/v2). The router's own prefix is /knowledge, so
    # the effective paths become /api/v1/knowledge/... — same shape the
    # _GATED / _UNGATED constants below use.
    app.include_router(knowledge_routes.router, prefix="/api/v1")
    return app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _capabilities_unavailable_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default both singletons to unavailable; tests opt into availability
    per-capability via the fixtures below. Mirrors ``tests/unit/test_memory
    /test_search/test_validate_components.py``.
    """
    monkeypatch.setattr(
        embedding_accessor, "_capability", EmbeddingCapability(provider=None)
    )
    monkeypatch.setattr(rerank_accessor, "_capability", RerankCapability(provider=None))


@pytest.fixture
def embed_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embedding_accessor, "_capability", EmbeddingCapability(provider=object())
    )


@pytest.fixture
def rerank_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rerank_accessor, "_capability", RerankCapability(provider=object())
    )


@pytest.fixture
def _stub_read_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every read/delete/list service call with a benign stub.

    The tests in this module care only about whether the capability
    gate fires. Ungated endpoints must reach their handler even when
    providers are missing — but reaching the handler with no
    MemoryRoot / SQLite setup would 500. Stubbing keeps the assertion
    focused on the gate, not on end-to-end service behavior.
    """
    from everos.service import knowledge as knowledge_service

    async def _fake_delete_document(*args: Any, **kwargs: Any) -> Any:
        return knowledge_service.DeleteResult(doc_id="d_abcdef123456", deleted_topics=0)

    async def _fake_list_documents(*args: Any, **kwargs: Any) -> Any:
        return knowledge_service.DocumentListResult(
            documents=[], total=0, page=1, page_size=20
        )

    async def _fake_get_document(*args: Any, **kwargs: Any) -> Any:
        from everos.component.utils.datetime import get_utc_now

        now = get_utc_now()
        return knowledge_service.DocumentDetail(
            doc_id="d_abcdef123456",
            category_id="General",
            title="stub",
            summary="stub",
            source_name=None,
            source_type=None,
            original_file_path=None,
            topics=[],
            created_at=now,
            updated_at=now,
        )

    async def _fake_get_topic(*args: Any, **kwargs: Any) -> Any:
        from everos.component.utils.datetime import get_utc_now

        now = get_utc_now()
        return knowledge_service.TopicDetail(
            topic_id="d_abcdef123456_0",
            doc_id="d_abcdef123456",
            category_id="General",
            topic_name="stub",
            topic_path="stub",
            depth=0,
            summary="stub",
            content="",
            content_labels=[],
            parent_topic_id=None,
            children_topic_ids=[],
            created_at=now,
            updated_at=now,
        )

    async def _fake_list_categories(*args: Any, **kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(knowledge_routes, "delete_document", _fake_delete_document)
    monkeypatch.setattr(knowledge_routes, "list_documents", _fake_list_documents)
    monkeypatch.setattr(knowledge_routes, "get_document", _fake_get_document)
    monkeypatch.setattr(knowledge_routes, "get_topic", _fake_get_topic)
    monkeypatch.setattr(knowledge_routes, "list_categories", _fake_list_categories)


@pytest.fixture
def _stub_patch_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``patch_document`` to echo the requested title back.

    Mirrors the real service contract: a title change lands in
    ``updated_fields`` and the caller sees the mutation reflected in
    the response envelope. Keeps the assertion focused on the gate
    (or lack of it) instead of on md / SQLite plumbing.
    """
    from everos.component.utils.datetime import get_utc_now
    from everos.service import knowledge as knowledge_service

    captured: dict[str, Any] = {}

    async def _fake_patch_document(
        doc_id: str,
        app_id: str,
        project_id: str,
        *,
        title: str | None = None,
        category_id: str | None = None,
    ) -> Any:
        captured["title"] = title
        captured["category_id"] = category_id
        updated_fields: list[str] = []
        if title is not None:
            updated_fields.append("title")
        if category_id is not None:
            updated_fields.append("category_id")
        return knowledge_service.PatchResult(
            doc_id=doc_id,
            updated_fields=updated_fields,
            updated_at=get_utc_now(),
        )

    monkeypatch.setattr(knowledge_routes, "patch_document", _fake_patch_document)


async def _call(client: AsyncClient, method: str, path: str) -> Any:
    return await client.request(method, path)


# ── Gated endpoints: 422 when embed missing (rerank also missing) ────────


@pytest.mark.parametrize("method,path", _GATED_ENDPOINTS)
async def test_gated_endpoint_422_when_embed_missing(
    client: AsyncClient, method: str, path: str
) -> None:
    """Write/search endpoints must 422 with an embedding-specific message
    when embedding is missing (check-embedding-first order per Task 12/13).
    """
    resp = await _call(client, method, path)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "embedding" in body["error"]["message"]
    assert "knowledge" in body["error"]["message"]


# ── Gated endpoints: 422 when only rerank missing ───────────────────────


@pytest.mark.parametrize("method,path", _GATED_ENDPOINTS)
async def test_gated_endpoint_422_when_rerank_missing(
    client: AsyncClient, method: str, path: str, embed_available: None
) -> None:
    """Write/search endpoints must 422 with a rerank-specific message
    when only rerank is missing.
    """
    resp = await _call(client, method, path)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "rerank" in body["error"]["message"]
    assert "knowledge" in body["error"]["message"]


# ── Ungated endpoints: must NOT 422 even with both capabilities missing ─


@pytest.mark.parametrize("method,path", _UNGATED_ENDPOINTS)
async def test_ungated_endpoint_not_422_without_any_capability(
    client: AsyncClient,
    method: str,
    path: str,
    _stub_read_services: None,
) -> None:
    """Read/list/delete endpoints must stay reachable even when neither
    embed nor rerank is configured — a Tier-3 → Tier-1 downgrade must
    not lock users out of inspecting and cleaning up their own state.
    """
    resp = await _call(client, method, path)
    assert resp.status_code != 422, (
        f"{method} {path} returned 422 despite the gate being scoped away "
        f"from read/delete endpoints; body={resp.text}"
    )


# ── Ungated endpoints: also fine with only embed (Tier-2 downgrade) ─────


@pytest.mark.parametrize("method,path", _UNGATED_ENDPOINTS)
async def test_ungated_endpoint_not_422_with_embed_only(
    client: AsyncClient,
    method: str,
    path: str,
    embed_available: None,
    _stub_read_services: None,
) -> None:
    """Tier-2 scenario (embed present, rerank missing): read/delete paths
    must stay reachable — this is the concrete regression from the
    router-wide-gate era that this fix targets.
    """
    resp = await _call(client, method, path)
    assert resp.status_code != 422, (
        f"{method} {path} returned 422 despite rerank not being needed; "
        f"body={resp.text}"
    )


# ── Gated endpoints: pass the gate when both are available ──────────────


async def test_search_not_422_when_both_available(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    embed_available: None,
    rerank_available: None,
) -> None:
    """Both capabilities available: the gate must not fire on /search.
    Stub the service call so the assertion is about the gate, not the
    rest of the search pipeline (out of scope for this task).
    """
    from everos.service import knowledge as knowledge_service

    stub_result = knowledge_service.SearchKnowledgeResult(hits=[], total=0, took_ms=0.0)

    async def _fake_search_knowledge(**kwargs: Any) -> Any:
        return stub_result

    monkeypatch.setattr(knowledge_routes, "search_knowledge", _fake_search_knowledge)
    resp = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "hello"},
    )
    assert resp.status_code != 422


# ── PATCH: metadata-only, must succeed regardless of provider config ────


_PATCH_BODY: dict[str, str] = {
    "app_id": "app1",
    "project_id": "proj1",
    "title": "renamed",
}


async def test_knowledge_patch_document_succeeds_without_rerank(
    client: AsyncClient,
    embed_available: None,
    _stub_patch_document: None,
) -> None:
    """Tier-2 scenario (embed configured, rerank missing): PATCH must
    still land. Renaming an existing document does not touch rerank —
    it only rewrites md frontmatter and upserts SQLite.
    """
    resp = await client.patch(
        "/api/v1/knowledge/documents/d_abcdef123456",
        json=_PATCH_BODY,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"]["doc_id"] == "d_abcdef123456"
    assert "title" in body["data"]["updated_fields"]


async def test_knowledge_patch_document_succeeds_without_any_capability(
    client: AsyncClient,
    _stub_patch_document: None,
) -> None:
    """Tier-1 scenario (no embed, no rerank): PATCH must still land.
    Metadata-only edits are a cleanup verb — locking users out of
    renaming their own docs after a full downgrade contradicts the
    per-endpoint-gate contract documented on
    ``_require_knowledge_capabilities``.
    """
    resp = await client.patch(
        "/api/v1/knowledge/documents/d_abcdef123456",
        json=_PATCH_BODY,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"]["doc_id"] == "d_abcdef123456"
    assert "title" in body["data"]["updated_fields"]
