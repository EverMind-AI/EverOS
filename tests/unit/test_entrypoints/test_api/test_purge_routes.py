"""The purge routes: validation as Cloud words it, status codes, DTO shape."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from everos.core.errors import NotFoundError, PurgeNonceInvalidError, StorageError
from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.schemas import (
    PurgeJobData,
    PurgeRequestData,
    PurgeStepData,
    PurgeSummary,
)
from everos.service.purge import (
    PurgeJobView,
    PurgeRequestResult,
    PurgeStep,
)
from everos.service.purge import PurgeSummary as SummaryView
from tests.fakes import StubPurgeService, make_application

NOW = datetime(2026, 11, 11, 10, 0, tzinfo=UTC)
JOB = PurgeJobView(
    job_id="pj_1",
    status="queued",
    scope="project",
    app_id="a1",
    project_id="p1",
    created_at=NOW,
    started_at=None,
    heartbeat_at=NOW,
    finished_at=None,
    steps=(PurgeStep("inbox", "pending"), PurgeStep("lancedb", "done", 3)),
)


@pytest.fixture
def purge() -> StubPurgeService:
    stub = StubPurgeService()
    stub.confirm.return_value = JOB
    stub.get.return_value = JOB
    stub.request.return_value = PurgeRequestResult(
        nonce="n" * 32,
        expires_at=NOW,
        scope="project",
        app_id="a1",
        project_id="p1",
        summary=SummaryView(
            markdown={"episode": 2},
            markdown_total=2,
            sqlite={"memcell": 1},
            sqlite_total=1,
            index_store="lancedb",
            index_total=4,
            complete=True,
            counted_at=NOW,
            comment="c",
        ),
    )
    return stub


@pytest.fixture
async def client(purge: StubPurgeService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(purge=purge))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


@pytest.mark.parametrize(
    ("body", "phrase"),
    [
        (
            {"scope": "space", "app_id": "a1"},
            "scope=space takes neither app_id nor project_id",
        ),
        (
            {"scope": "space", "project_id": "p1"},
            "scope=space takes neither app_id nor project_id",
        ),
        ({"scope": "app"}, "scope=app requires app_id"),
        (
            {"scope": "app", "app_id": "a1", "project_id": "p1"},
            "scope=app takes no project_id",
        ),
        (
            {"scope": "project", "app_id": "a1"},
            "scope=project requires app_id and project_id",
        ),
        (
            {"scope": "project", "project_id": "p1"},
            "scope=project requires app_id and project_id",
        ),
        ({"scope": "tenant"}, None),
        ({"scope": "space", "user_id": "u1"}, None),
        ({"scope": "app", "app_id": "default_app"}, None),
        ({"scope": "app", "app_id": "../x"}, None),
    ],
)
async def test_request_validation_matrix(
    client: AsyncClient, purge: StubPurgeService, body: dict, phrase: str | None
) -> None:
    for version in ("v1", "v2"):
        resp = await client.post(f"/api/{version}/memory/purge", json=body)
        assert resp.status_code == 422, resp.text
        if phrase is not None:
            assert phrase in resp.json()["error"]["message"]
    purge.request.assert_not_awaited()


async def test_request_returns_summary_under_the_backend_field(
    client: AsyncClient, purge: StubPurgeService
) -> None:
    resp = await client.post("/api/v2/memory/purge", json={"scope": "space"})
    assert resp.status_code == 200
    summary = resp.json()["data"]["summary"]
    assert (summary["lancedb_total"], summary["milvus_total"]) == (4, None)
    assert summary["markdown"] == {"episode": 2} and summary["complete"] is True
    purge.request.assert_awaited_once_with("space", None, None)


@pytest.mark.parametrize(
    "body",
    [
        {"nonce": "short"},
        {"nonce": "x" * 16 + "!"},
        {"nonce": "x" * 65},
        {"nonce": "x" * 20, "scope": "space"},
        {},
    ],
)
async def test_malformed_nonce_is_422(
    client: AsyncClient, purge: StubPurgeService, body: dict
) -> None:
    resp = await client.post("/api/v2/memory/purge/confirm", json=body)
    assert resp.status_code == 422
    purge.confirm.assert_not_awaited()


async def test_confirm_returns_202_job_data(
    client: AsyncClient, purge: StubPurgeService
) -> None:
    resp = await client.post("/api/v2/memory/purge/confirm", json={"nonce": "n" * 32})
    assert resp.status_code == 202
    data = resp.json()["data"]
    assert (data["job_id"], data["status"], data["scope"]) == (
        "pj_1",
        "queued",
        "project",
    )
    assert data["steps"][1] == {
        "store": "lancedb",
        "state": "done",
        "deleted": 3,
        "error": None,
    }
    purge.confirm.assert_awaited_once_with("n" * 32)


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (PurgeNonceInvalidError("bad"), 404, "PURGE_NONCE_INVALID"),
        (
            StorageError("could not record the purge job; retry"),
            503,
            "EXTERNAL_SERVICE_UNAVAILABLE",
        ),
    ],
)
async def test_confirm_errors(
    client: AsyncClient,
    purge: StubPurgeService,
    error: Exception,
    status: int,
    code: str,
) -> None:
    purge.confirm.side_effect = error
    resp = await client.post("/api/v2/memory/purge/confirm", json={"nonce": "n" * 32})
    assert (resp.status_code, resp.json()["error"]["code"]) == (status, code)


async def test_get_job(client: AsyncClient, purge: StubPurgeService) -> None:
    resp = await client.get("/api/v1/memory/purge/pj_1")
    assert resp.status_code == 200 and resp.json()["data"]["job_id"] == "pj_1"
    purge.get.side_effect = NotFoundError("purge job 'pj_x' not found")
    resp = await client.get("/api/v2/memory/purge/pj_x")
    assert (resp.status_code, resp.json()["error"]["code"]) == (404, "NOT_FOUND")
    assert (await client.get("/api/v2/memory/purge/bad!id")).status_code == 422


def test_dto_fields_match_cloud() -> None:
    """Cloud ``purge_api.py`` (main) field for field; stores renamed (D7)."""
    assert list(PurgeJobData.model_fields) == [
        "job_id",
        "status",
        "scope",
        "app_id",
        "project_id",
        "created_at",
        "started_at",
        "heartbeat_at",
        "finished_at",
        "steps",
    ]
    assert list(PurgeStepData.model_fields) == ["store", "state", "deleted", "error"]
    assert list(PurgeRequestData.model_fields) == [
        "nonce",
        "expires_at",
        "scope",
        "app_id",
        "project_id",
        "summary",
    ]
    assert list(PurgeSummary.model_fields) == [
        "markdown",
        "markdown_total",
        "sqlite",
        "sqlite_total",
        "lancedb_total",
        "milvus_total",
        "complete",
        "counted_at",
        "comment",
        "error",
    ]


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (None, True),  # curl, SDKs, MCP, plugins: no Origin
        ("http://t", True),  # a page served by this server itself
        ("HTTP://T", True),
        ("https://evil.example", False),
        ("http://t:8001", False),  # same host, another port: another site
        ("null", False),  # a sandboxed or file:// page
    ],
)
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v2/memory/purge", {"scope": "space"}),
        ("/api/v1/memory/purge/confirm", {"nonce": "n" * 32}),
    ],
)
async def test_a_browser_page_of_another_site_cannot_purge(
    client: AsyncClient,
    purge: StubPurgeService,
    origin: str | None,
    allowed: bool,
    path: str,
    body: dict[str, str],
) -> None:
    headers = {"origin": origin} if origin is not None else {}
    resp = await client.post(path, json=body, headers=headers)
    called = purge.request.await_count + purge.confirm.await_count
    if allowed:
        assert resp.status_code in (200, 202), resp.text
        assert called == 1
    else:
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "CROSS_SITE_REFUSED"
        assert called == 0
