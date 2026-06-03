"""422 validation paths for ``POST /api/v1/memory/get``.

These are route-layer error tests — they exercise:

- DTO-layer rejections (page_size cap, empty owner_id, missing /
  invalid memory_type, invalid sort_order, owner+memory_type mismatch)
- service-layer ``compile_filters_for_get`` rejections (unknown filter
  field, malformed op shape)

No data is seeded; nothing reaches LanceDB. The full happy-path / data
e2e suite (with seeded rows and 200 assertions) lives in
``tests/integration/test_get_endpoint_e2e.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from importlib import import_module
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.infra.persistence.lancedb import lancedb_manager

# ``everos.service.__init__`` re-exports ``get`` shadowing the
# submodule. Reach the real module via importlib so we can reset its
# ``_manager`` lazy singleton.
get_service_mod = import_module("everos.service.get")


@pytest.fixture
async def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """FastAPI app with no lifespan; resets get-path singletons per test."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    load_settings.cache_clear()

    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    get_service_mod._manager = None

    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await lancedb_manager.dispose_connection()
    load_settings.cache_clear()


# ── DTO-layer 422 ──────────────────────────────────────────────────────


async def test_page_size_above_cap_returns_422(client: AsyncClient) -> None:
    """``page_size > 100`` violates the wiki cap → 422 at the DTO layer."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "page_size": 200,
        },
    )
    assert resp.status_code == 422


async def test_empty_user_id_returns_422(client: AsyncClient) -> None:
    """``user_id`` carries ``min_length=1`` end-to-end."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "",
            "memory_type": "episode",
        },
    )
    assert resp.status_code == 422


async def test_missing_memory_type_returns_422(client: AsyncClient) -> None:
    """Omitting the required ``memory_type`` field is rejected at the DTO layer."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={"user_id": "u1"},
    )
    assert resp.status_code == 422


async def test_invalid_memory_type_value_returns_422(client: AsyncClient) -> None:
    """``memory_type`` outside the four-kind enum → 422."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "atomic_fact",  # not a top-level kind
        },
    )
    assert resp.status_code == 422


async def test_invalid_sort_order_returns_422(client: AsyncClient) -> None:
    """``sort_order`` is a tight Literal — uppercase variant rejected."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "sort_order": "DESC",
        },
    )
    assert resp.status_code == 422


async def test_owner_memory_type_mismatch_returns_422(client: AsyncClient) -> None:
    """``user`` + ``agent_case`` is a hard pydantic error."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "agent_case",
        },
    )
    assert resp.status_code == 422


# ── service.compile_filters_for_get 422 ───────────────────────────────


async def test_unknown_filter_field_returns_422(client: AsyncClient) -> None:
    """A field outside ``ALLOWED_FIELDS`` surfaces as 422 from the adapter."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"random_attr": "boom"},
        },
    )
    assert resp.status_code == 422
    assert "unsupported" in resp.text


async def test_malformed_filter_in_op_returns_422(client: AsyncClient) -> None:
    """``in`` op with a scalar (not list) surfaces as 422 from the adapter."""
    resp = await client.post(
        "/api/v1/memory/get",
        json={
            "user_id": "u1",
            "memory_type": "episode",
            "filters": {"session_id": {"in": "not_a_list"}},
        },
    )
    assert resp.status_code == 422
