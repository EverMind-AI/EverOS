"""While a purge runs every write route answers 409; every read passes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse

from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.middleware import PurgeGateMiddleware
from everos.entrypoints.api.middleware.purge_gate import READ_ROUTES, WRITE_ROUTES
from tests.fakes import StubPurgeService, make_application

_IDS = {
    "{doc_id}": "d_abcdef123456",
    "{topic_id}": "d_abcdef123456_1",
    "{task_id}": "t1",
    "{job_id}": "pj_1",
}


_SPEC_READS = {
    ("POST", "/memory/search"),
    ("POST", "/memory/get"),
    ("POST", "/knowledge/search"),
    ("POST", "/memory/purge"),
    ("POST", "/memory/purge/confirm"),
}
"""The non-GET reads spec §5.2 names, written out independently of the table."""


def _concrete(path: str) -> str:
    for placeholder, value in _IDS.items():
        path = path.replace(placeholder, value)
    return path


def _versions(path: str) -> list[str]:
    unversioned = ("/health", "/metrics", "/docs", "/redoc", "/openapi.json")
    if path.startswith(unversioned):
        return [path]
    return [f"/api/v1{path}", f"/api/v2{path}"]


@pytest.fixture
def purge() -> StubPurgeService:
    stub = StubPurgeService()
    stub.active = "pj_running"
    return stub


@pytest.fixture
async def client(purge: StubPurgeService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(purge=purge))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_writes_409_reads_pass(client: AsyncClient) -> None:
    for method, template in sorted(WRITE_ROUTES):
        for path in _versions(_concrete(template)):
            resp = await client.request(method, path, json={})
            assert resp.status_code == 409, (method, path)
            error = resp.json()["error"]
            assert error["code"] == "PURGE_IN_PROGRESS"
            assert "pj_running" in error["message"]
            assert "GET /api/v2/memory/purge/pj_running" in error["message"]
            assert resp.json()["request_id"] == resp.headers["X-Request-Id"]


async def test_reads_pass_the_gate(purge: StubPurgeService) -> None:
    """The gate alone, over an app answering 200 to anything: every read
    route (both versions) reaches it while a purge runs."""

    async def ok(scope: Any, receive: Any, send: Any) -> None:
        await PlainTextResponse("ok")(scope, receive, send)

    inner = Starlette()
    inner.state.application = SimpleNamespace(purge=purge)
    gate = PurgeGateMiddleware(ok)

    async def app(scope: Any, receive: Any, send: Any) -> None:
        scope["app"] = inner
        await gate(scope, receive, send)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for method, template in sorted(READ_ROUTES | _SPEC_READS):
            for path in _versions(_concrete(template)):
                resp = await c.request(method, path, json={})
                assert resp.status_code == 200, (method, path)
        assert (await c.post("/api/v2/memory/add", json={})).status_code == 409


async def test_writes_pass_without_a_purge(
    client: AsyncClient, purge: StubPurgeService
) -> None:
    purge.active = None
    resp = await client.post("/api/v2/memory/add", json={})
    assert resp.status_code == 422  # reached the route's validation


async def test_docs_and_options_pass_during_purge(client: AsyncClient) -> None:
    preflight = await client.options(
        "/api/v2/memory/add",
        headers={
            "Origin": "http://ui.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.status_code == 200
    assert (await client.get("/health")).status_code == 200


async def test_409_carries_cors_headers(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v2/memory/add", json={}, headers={"Origin": "http://ui.example"}
    )
    assert resp.status_code == 409
    assert resp.headers.get("access-control-allow-origin") in ("*", "http://ui.example")


async def test_unclassified_write_is_held(client: AsyncClient) -> None:
    """A route nobody classified is refused like a write (R7); GET passes."""
    assert (await client.post("/api/v2/memory/nope", json={})).status_code == 409
    assert (await client.get("/api/v2/memory/nope")).status_code == 404
