"""``POST /memory/delete`` at the route: validation, envelope, error codes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from everos.core.errors import DeletionIncompleteError, SessionBusyError
from everos.entrypoints.api.app import create_app
from everos.service.memory import DeleteResult
from tests.fakes import StubMemoryService, make_application


@pytest.fixture
def memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.delete.return_value = DeleteResult(filters=["session_id"], count=3)
    return stub


@pytest.fixture
async def client(memory: StubMemoryService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"app_id": "a"},
        {"user_id": "u1", "agent_id": "a1"},
        {"user_id": "u1", "agent_id": "a1", "session_id": "s"},
        {"user_id": "a/b"},
        {"user_id": ".."},
        {"user_id": "."},
        {"agent_id": "a\\b"},
        {"user_id": "a∕b"},
        {"user_id": "a%2Fb"},
        {"user_id": ""},
        {"session_id": ""},
        {"session_id": "s", "extra": 1},
        {"session_id": "s", "app_id": "default_app"},
    ],
)
async def test_invalid_scope_is_422_and_deletes_nothing(
    client: AsyncClient, memory: StubMemoryService, body: dict[str, Any]
) -> None:
    """A2 / RF4: no scope, both owners, a path-like id, an unknown field."""
    resp = await client.post("/api/v2/memory/delete", json=body)
    assert resp.status_code == 422, resp.text
    memory.delete.assert_not_awaited()


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_ok_answers_filters_and_count_in_the_envelope(
    client: AsyncClient, memory: StubMemoryService, version: str
) -> None:
    resp = await client.post(
        f"/api/{version}/memory/delete", json={"session_id": "s1", "project_id": "p"}
    )

    assert resp.status_code == 200
    assert resp.json()["data"] == {"filters": ["session_id"], "count": 3}
    memory.delete.assert_awaited_once_with(
        app_id="default", project_id="p", user_id=None, agent_id=None, session_id="s1"
    )


async def test_session_busy_is_503_with_its_code(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    memory.delete.side_effect = SessionBusyError(
        "session is busy; nothing was deleted, retry later"
    )

    resp = await client.post("/api/v2/memory/delete", json={"session_id": "s1"})

    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["code"] == "SESSION_BUSY"
    assert "nothing was deleted, retry later" in error["message"]


async def test_incomplete_is_500_asking_for_the_same_request(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    memory.delete.side_effect = DeletionIncompleteError("disk full")

    resp = await client.post("/api/v2/memory/delete", json={"user_id": "u1"})

    assert resp.status_code == 500
    error = resp.json()["error"]
    assert error["code"] == "INTERNAL_ERROR"
    assert error["message"] == "deletion incomplete; retry the same request"


# ── /delete_by_ids ───────────────────────────────────────────────────────

_OK_ID = "u1_ep_20250514_0001"


@pytest.mark.parametrize(
    "body",
    [
        {"memory_type": "atomic_fact", "memory_ids": [_OK_ID]},
        {"memory_ids": [_OK_ID]},
        {"memory_type": "episode", "memory_ids": []},
        {"memory_type": "episode", "memory_ids": [_OK_ID] * 51},
        {"memory_type": "episode", "memory_ids": [_OK_ID, "_ep_20250514_0001"]},
        {"memory_type": "episode", "memory_ids": [_OK_ID, "u1_ep_20251399_0001"]},
        {"memory_type": "episode", "memory_ids": [_OK_ID, "a/b_ep_20250514_0001"]},
        {"memory_type": "episode", "memory_ids": [_OK_ID, ".._ep_20250514_0001"]},
        {"memory_type": "episode", "memory_ids": [_OK_ID, "u1_af_20250514_0001"]},
        {"memory_type": "episode", "memory_ids": [_OK_ID], "reason": {"code": "nope"}},
        {
            "memory_type": "episode",
            "memory_ids": [_OK_ID],
            "reason": {"code": "other", "note": "x" * 257},
        },
        {"memory_type": "episode", "memory_ids": [_OK_ID], "extra": True},
    ],
)
async def test_by_ids_invalid_request_is_422_and_deletes_nothing(
    client: AsyncClient, memory: StubMemoryService, body: dict[str, Any]
) -> None:
    """B1: one bad id rejects the whole request, valid ids included."""
    resp = await client.post("/api/v2/memory/delete_by_ids", json=body)
    assert resp.status_code == 422, resp.text
    memory.delete_by_ids.assert_not_awaited()


async def test_by_ids_ok_dedupes_and_answers_cloud_shape(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    other = "u.x@y+z_ep_20240229_0012"  # an owner with dots, @, +; a leap day
    resp = await client.post(
        "/api/v1/memory/delete_by_ids",
        json={
            "memory_type": "episode",
            "memory_ids": [_OK_ID, other, _OK_ID],
            "reason": {"code": "privacy", "note": "n"},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["data"] == {"memory_type": "episode", "success": True}
    memory.delete_by_ids.assert_awaited_once_with(
        app_id="default",
        project_id="default",
        memory_type="episode",
        ids=[_OK_ID, other],
        reason_code="privacy",
        note="n",
    )
