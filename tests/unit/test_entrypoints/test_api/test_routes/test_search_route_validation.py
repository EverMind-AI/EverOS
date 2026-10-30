"""422 validation paths for ``POST /api/v1/memory/search``.

These exercise the request → DTO / route → service.compile_filters
error paths *without* needing any seeded data or external services
(no embedder / no LLM / no LanceDB rows). The full data-driven e2e
suite lives in ``tests/integration/test_search_endpoint_e2e.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.service import build_application
from tests.fakes import make_providers
from tests.fakes.storage import Storage


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[AsyncClient]:
    """FastAPI app over a fresh, unstarted application per test."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()

    app = create_app(
        application=build_application(load_settings(), providers=make_providers())
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await storage.lance.close()
    load_settings.cache_clear()


def _body(**overrides) -> dict:
    """Minimal valid SearchRequest body; tests override one field to break it.

    ``method="keyword"`` is pinned because the SearchRequest DTO defaults
    to HYBRID, which ``SearchManager._validate_components`` rejects when
    no ``[embedding]`` provider is configured (the case in CI). Keyword
    needs no embedder, so DTO / compile_filters validation paths fire
    cleanly without external services — which is exactly what this file
    is supposed to exercise.
    """
    base = {
        "user_id": "u1",
        "query": "hello",
        "method": "keyword",
    }
    base.update(overrides)
    return base


# ── DTO-layer 422 ──────────────────────────────────────────────────────


async def test_empty_query_returns_422(client: AsyncClient) -> None:
    """``query`` carries ``min_length=1``."""
    resp = await client.post("/api/v1/memory/search", json=_body(query=""))
    assert resp.status_code == 422


async def test_empty_user_id_returns_422(client: AsyncClient) -> None:
    """``user_id`` carries ``min_length=1``."""
    resp = await client.post("/api/v1/memory/search", json=_body(user_id=""))
    assert resp.status_code == 422


async def test_both_user_and_agent_id_returns_422(client: AsyncClient) -> None:
    """Both ``user_id`` and ``agent_id`` set → xor validator rejects."""
    resp = await client.post("/api/v1/memory/search", json=_body(agent_id="agent_x"))
    assert resp.status_code == 422


async def test_invalid_method_returns_422(client: AsyncClient) -> None:
    """``method`` outside the SearchMethod enum → 422."""
    resp = await client.post("/api/v1/memory/search", json=_body(method="bm42"))
    assert resp.status_code == 422


async def test_top_k_zero_returns_422(client: AsyncClient) -> None:
    """``top_k=0`` violates the validator (must be -1 or 1..100)."""
    resp = await client.post("/api/v1/memory/search", json=_body(top_k=0))
    assert resp.status_code == 422


async def test_top_k_above_cap_returns_422(client: AsyncClient) -> None:
    """``top_k=101`` exceeds the 100 cap."""
    resp = await client.post("/api/v1/memory/search", json=_body(top_k=101))
    assert resp.status_code == 422


async def test_radius_above_one_returns_422(client: AsyncClient) -> None:
    """``radius`` is constrained to [0.0, 1.0]."""
    resp = await client.post("/api/v1/memory/search", json=_body(radius=1.5))
    assert resp.status_code == 422


# ── service.compile_filters 422 ───────────────────────────────────────


async def test_unknown_filter_field_returns_422(client: AsyncClient) -> None:
    """A field outside ``ALLOWED_FIELDS`` surfaces as 422 from the adapter."""
    resp = await client.post(
        "/api/v1/memory/search",
        json=_body(filters={"random_attr": "boom"}),
    )
    assert resp.status_code == 422
    assert "unsupported" in resp.text


async def test_reserved_owner_id_in_filters_returns_422(client: AsyncClient) -> None:
    """``owner_id`` is reserved at the top level — must not appear inside filters."""
    resp = await client.post(
        "/api/v1/memory/search",
        json=_body(filters={"owner_id": "spoof"}),
    )
    assert resp.status_code == 422
