"""A write into the range a purge is clearing answers 409 ``PURGE_IN_PROGRESS``.

Each refusing route asks the purge about the app / project of its own
request; every other non-GET route is listed with why it is not refused, so
a new write route fails here until someone decides.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from everos.core.errors import PurgeInProgressError
from everos.entrypoints.api.app import create_app
from tests.fakes import StubPurgeService, make_application

WHERE = {"app_id": "a1", "project_id": "p1"}
DOC = "d_" + "a" * 12
_MESSAGE = {"sender_id": "u1", "role": "user", "timestamp": 1, "content": "hi"}
_UPLOAD = {"file": ("n.md", b"# n\n", "text/markdown")}

REFUSED: dict[tuple[str, str], dict[str, Any]] = {
    ("POST", "/memory/add"): {"json": {"session_id": "s", "messages": [_MESSAGE]}},
    ("POST", "/memory/flush"): {"json": {"session_id": "s"}},
    ("POST", "/knowledge/documents"): {"data": {"title": "n"}, "files": _UPLOAD},
    ("PUT", "/knowledge/documents/{doc_id}"): {
        "data": {"title": "n"},
        "files": _UPLOAD,
    },
    ("PATCH", "/knowledge/documents/{doc_id}"): {"json": {"title": "n"}},
}
EXEMPT: dict[tuple[str, str], str] = {
    ("POST", "/memory/search"): "a read",
    ("POST", "/memory/get"): "a read",
    ("POST", "/knowledge/search"): "a read",
    ("POST", "/memory/delete"): "removes data, as the purge does",
    ("POST", "/memory/delete_by_ids"): "removes data, as the purge does",
    ("DELETE", "/knowledge/documents/{doc_id}"): "removes data, as the purge does",
    ("POST", "/memory/purge"): "the purge itself",
    ("POST", "/memory/purge/confirm"): "the purge itself",
    ("POST", "/ome/trigger"): "names no app / project (known gap G1)",
    ("POST", "/cascade/quiesce"): "names no app / project",
}


def _write_routes() -> set[tuple[str, str]]:
    paths = create_app(application=make_application()).openapi()["paths"]
    return {
        (method.upper(), path.removeprefix("/api/v1").removeprefix("/api/v2"))
        for path, ops in paths.items()
        for method in ops
        if method not in ("get", "head", "options")
    }


def test_every_non_get_route_is_refused_or_exempt() -> None:
    assert _write_routes() == set(REFUSED) | set(EXEMPT)


@pytest.fixture
def purge() -> StubPurgeService:
    stub = StubPurgeService()
    stub.refusing = PurgeInProgressError("purge job pj_1 is clearing this range")
    return stub


@pytest.fixture
async def client(purge: StubPurgeService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(purge=purge))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


@pytest.mark.parametrize(("method", "path"), sorted(REFUSED))
async def test_a_write_into_the_purged_range_is_409(
    client: AsyncClient, purge: StubPurgeService, method: str, path: str
) -> None:
    kw = REFUSED[(method, path)]
    if "json" in kw:
        kw = {"json": {**kw["json"], **WHERE}}
    else:
        kw = {**kw, "data": {**kw["data"], **WHERE}}
    url = "/api/v1" + path.replace("{doc_id}", DOC)
    resp = await client.request(method, url, **kw)
    assert resp.status_code == 409, resp.text
    error = resp.json()["error"]
    assert error["code"] == "PURGE_IN_PROGRESS"
    assert "pj_1" in error["message"]
    assert purge.checked_writes == [("a1", "p1")]
