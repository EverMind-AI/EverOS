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
from everos.service import NO_AGENT_MEMORY, build_application
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

    No embedder is configured (the case in CI), so a valid body would be
    refused by the manager's component guard -- after the DTO and the
    filters, which are what this file exercises.
    """
    base = {
        "user_id": "u1",
        "query": "hello",
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


async def test_an_agent_scope_returns_422_saying_why(client: AsyncClient) -> None:
    """EverOS 2.0 has no agent memory: an ``agent_id`` (or a
    ``context_group_id``) is refused with a message that says so."""
    for body in (
        _body(agent_id="agent_x"),
        {"agent_id": "agent_x", "query": "q"},
        _body(context_group_id="g1"),
    ):
        resp = await client.post("/api/v1/memory/search", json=body)
        assert resp.status_code == 422
        assert NO_AGENT_MEMORY in resp.json()["error"]["message"]


@pytest.mark.parametrize("method", ["keyword", "vector", "hybrid", "agentic", "bm42"])
async def test_a_removed_method_returns_422_naming_multiround(
    client: AsyncClient, method: str
) -> None:
    """Every method but ``llm_multiround`` is refused with a 4xx that says
    multi-round retrieval is the only one."""
    resp = await client.post("/api/v2/memory/search", json=_body(method=method))
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "INVALID_INPUT"
    assert "multi-round retrieval only" in error["message"]
    assert "'llm_multiround'" in error["message"]
    assert repr(method) in error["message"]


async def test_without_an_embedder_a_valid_search_is_answered(
    client: AsyncClient,
) -> None:
    """No embedder is a keyword-only deployment, not a refused search."""
    resp = await client.post(
        "/api/v2/memory/search", json=_body(method="llm_multiround")
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["episodes"] == []


async def test_top_k_zero_returns_422(client: AsyncClient) -> None:
    """``top_k=0`` violates the validator (must be -1 or 1..100)."""
    resp = await client.post("/api/v1/memory/search", json=_body(top_k=0))
    assert resp.status_code == 422


async def test_top_k_above_cap_returns_422(client: AsyncClient) -> None:
    """``top_k=101`` exceeds the 100 cap."""
    resp = await client.post("/api/v1/memory/search", json=_body(top_k=101))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "option", [{"radius": 0.5}, {"min_score": 0.4}, {"enable_llm_rerank": True}]
)
async def test_the_removed_methods_options_return_422(
    client: AsyncClient, option: dict
) -> None:
    """``radius`` / ``min_score`` / ``enable_llm_rerank`` tuned the removed
    methods; the request no longer has them."""
    resp = await client.post("/api/v1/memory/search", json=_body(**option))
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
