"""``/memory/update``, ``/memory/history``, and ``/memory/forget`` by target.

Protocol translation only; the service is a stub. What these pin:

- an update reaches the service with its scope, id, expected version, the
  fields it changes and its hold; a stale or forgotten memory answers 409 with
  its own error code; a body naming both or neither owner is 422;
- a history read takes the id and one owner as query parameters and answers
  the versions with their times as datetimes;
- ``/memory/forget`` routes three bodies by shape -- a user's target, the
  picture body, a group's -- and the two target shapes are a dry run unless
  ``dry_run`` is false; ``/memory/forget/undo`` routes a user's ``forget_id``,
  a picture's body and a group's;
- malformed targets are 422 before the service is called;
- ``/memory/delete`` and ``/delete_by_ids`` answer with the forgets' ids.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from everos.core.errors import ForgottenMemoryError, VersionConflictError
from everos.entrypoints.api.app import create_app
from everos.service import (
    GroupForgetResult,
    GroupUndoResult,
    MemoryHistory,
    MemoryVersion,
    UpdateResult,
    UserForgetResult,
    UserForgetTarget,
    UserUndoResult,
)
from everos.service.memory import DeleteResult, ForgetResult, UndoResult
from tests.fakes import StubMemoryService, make_application

_HASH = "ab" * 32
_FORGET_ID = "fg_0123456789abcdef"
T0 = 1_772_323_200_000


@pytest.fixture
def memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.update_memory.return_value = UpdateResult(
        kind="episode",
        id="amy_ep_20261112_0002",
        supersedes="amy_ep_20261112_0001",
        changed=True,
        held=True,
        edited_by="amy",
        edited_at=T0,
    )
    stub.memory_history.return_value = MemoryHistory(
        kind="episode",
        id="amy_ep_20261112_0001",
        current="amy_ep_20261112_0002",
        versions=(
            MemoryVersion(
                id="amy_ep_20261112_0002",
                state="current",
                timestamp=T0,
                supersedes="amy_ep_20261112_0001",
                edited_by="amy",
                edited_at=T0 + 5,
                held=True,
            ),
            MemoryVersion(
                id="amy_ep_20261112_0001",
                state="superseded",
                timestamp=T0,
                next="amy_ep_20261112_0002",
            ),
        ),
    )
    stub.forget.side_effect = lambda **kw: UserForgetResult(
        forget_id="" if kw["dry_run"] else _FORGET_ID,
        dry_run=kw["dry_run"],
        target=kw["target"],
        episodes=("amy_ep_20261112_0001",),
    )
    stub.undo_forget.return_value = UserUndoResult(
        forget_id=_FORGET_ID,
        applied=True,
        target=UserForgetTarget(kind="owner"),
        episodes=("amy_ep_20261112_0001",),
        profile_items=2,
    )
    stub.forget_in_group.side_effect = lambda **kw: GroupForgetResult(
        forget_id="" if kw["dry_run"] else _FORGET_ID,
        dry_run=kw["dry_run"],
        target=kw["target"],
    )
    stub.forget_image.return_value = ForgetResult(
        image_id=_HASH, image_memories=0, episodes=(), profiles=0
    )
    stub.undo_forget_image.return_value = UndoResult(
        image_id=_HASH,
        forget_id="",
        applied=False,
        image_memories=0,
        episodes=(),
        profiles=0,
    )
    stub.undo_group_forget.return_value = GroupUndoResult(
        forget_id=_FORGET_ID, applied=True
    )
    stub.delete.return_value = DeleteResult(
        filters=["session_id"], count=3, forget_ids=(_FORGET_ID,)
    )
    stub.delete_by_ids.return_value = (_FORGET_ID,)
    return stub


@pytest.fixture
async def client(memory: StubMemoryService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


# ── update ───────────────────────────────────────────────────────────────


async def test_an_update_reaches_the_service_and_answers_the_new_version(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/update",
        json={
            "user_id": "amy",
            "id": "amy_ep_20261112_0001",
            "content": "Amy adopted a dog.",
            "expected_version": "amy_ep_20261112_0001",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["id"] == "amy_ep_20261112_0002" and data["held"] is True
    assert data["supersedes"] == "amy_ep_20261112_0001"
    assert data["edited_at"].startswith("2026-03-01")
    kwargs = memory.update_memory.await_args.kwargs
    assert kwargs["user_id"] == "amy" and kwargs["group_id"] is None
    assert kwargs["record_id"] == "amy_ep_20261112_0001"
    assert kwargs["expected_version"] == "amy_ep_20261112_0001"
    assert kwargs["changes"].content == "Amy adopted a dog."
    assert kwargs["changes"].description is None and kwargs["hold"] is True


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            VersionConflictError("current is amy_ep_2", current="amy_ep_2"),
            "VERSION_CONFLICT",
        ),
        (ForgottenMemoryError("forgotten"), "MEMORY_FORGOTTEN"),
    ],
)
async def test_a_conflict_answers_409_with_its_code(
    client: AsyncClient, memory: StubMemoryService, error: Exception, code: str
) -> None:
    memory.update_memory.side_effect = error
    response = await client.post(
        "/api/v2/memory/update",
        json={"user_id": "amy", "id": "amy_ep_20261112_0001", "content": "x"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "body",
    [
        {"id": "amy_ep_1", "content": "x"},  # no owner
        {"user_id": "a", "group_id": "g", "id": "amy_ep_1"},  # both owners
        {"user_id": "a", "id": "../etc", "content": "x"},  # not an id
        {"user_id": "a", "id": "amy_ep_1", "content": ""},  # an empty body
        {"user_id": "a", "id": "amy_ep_1", "colour": "red"},  # an unknown field
    ],
)
async def test_a_malformed_update_is_422(
    client: AsyncClient, memory: StubMemoryService, body: dict[str, Any]
) -> None:
    response = await client.post("/api/v2/memory/update", json=body)
    assert response.status_code == 422
    memory.update_memory.assert_not_awaited()


async def test_a_group_update_passes_the_asker_and_the_identity_header(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/update",
        json={"group_id": "trip", "asker_id": "u_chen", "id": "pi_0", "hold": False},
    )
    assert response.status_code == 200
    kwargs = memory.update_memory.await_args.kwargs
    assert (kwargs["group_id"], kwargs["asker_id"], kwargs["hold"]) == (
        "trip",
        "u_chen",
        False,
    )


# ── history ──────────────────────────────────────────────────────────────


async def test_a_history_read_answers_every_version(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.get(
        "/api/v2/memory/history",
        params={"user_id": "amy", "id": "amy_ep_20261112_0001"},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["current"] == "amy_ep_20261112_0002"
    assert [v["state"] for v in data["versions"]] == ["current", "superseded"]
    assert data["versions"][1]["next"] == "amy_ep_20261112_0002"
    assert data["versions"][0]["edited_by"] == "amy"
    kwargs = memory.memory_history.await_args.kwargs
    assert kwargs["record_id"] == "amy_ep_20261112_0001"


async def test_a_history_read_names_exactly_one_owner(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    for params in ({"id": "x_ep_1"}, {"id": "x_ep_1", "user_id": "a", "group_id": "g"}):
        response = await client.get("/api/v2/memory/history", params=params)
        assert response.status_code == 422
    memory.memory_history.assert_not_awaited()


# ── forget ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            {"kind": "record", "record_ids": ["amy_ep_1", "pi_00"]},
            UserForgetTarget(kind="record", record_ids=("amy_ep_1", "pi_00")),
        ),
        (
            {"kind": "session", "session_id": "s1"},
            UserForgetTarget(kind="session", session_id="s1"),
        ),
        ({"kind": "owner"}, UserForgetTarget(kind="owner")),
        (
            {
                "kind": "condition",
                "memory_types": ["episode", "profile"],
                "start": 10,
                "end": 20,
            },
            UserForgetTarget(
                kind="condition",
                memory_types=("episode", "profile"),
                start=10,
                end=20,
            ),
        ),
        (
            {"kind": "image", "image_id": _HASH},
            UserForgetTarget(kind="image", image_id=_HASH),
        ),
    ],
)
async def test_a_users_forget_is_a_dry_run_by_default(
    client: AsyncClient,
    memory: StubMemoryService,
    target: dict[str, Any],
    expected: UserForgetTarget,
) -> None:
    response = await client.post(
        "/api/v2/memory/forget", json={"user_id": "amy", "target": target}
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["dry_run"] is True and data["forget_id"] == ""
    kwargs = memory.forget.await_args.kwargs
    assert kwargs["target"] == expected and kwargs["dry_run"] is True

    response = await client.post(
        "/api/v2/memory/forget",
        json={"user_id": "amy", "target": target, "dry_run": False},
    )
    assert response.json()["data"]["forget_id"] == _FORGET_ID


async def test_the_three_forget_bodies_reach_their_own_methods(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    await client.post(
        "/api/v2/memory/forget", json={"user_id": "amy", "image_id": _HASH}
    )
    memory.forget_image.assert_awaited_once()
    response = await client.post(
        "/api/v2/memory/forget",
        json={
            "group_id": "trip",
            "asker_id": "u_chen",
            "target": {"kind": "session", "session_id": "s1"},
        },
    )
    assert response.status_code == 200, response.text
    assert memory.forget_in_group.await_args.kwargs["dry_run"] is True
    memory.forget.assert_not_awaited()


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "condition", "memory_types": ["episode"]},  # no range
        {"kind": "condition", "memory_types": ["episode"], "start": 5, "end": 5},
        {"kind": "condition", "memory_types": ["agent_case"], "start": 1},
        {"kind": "record", "record_ids": []},
        {"kind": "member", "member_id": "u"},  # a group's target
        {"kind": "image", "image_id": "not-a-hash"},
    ],
)
async def test_a_malformed_user_target_is_422(
    client: AsyncClient, memory: StubMemoryService, target: dict[str, Any]
) -> None:
    response = await client.post(
        "/api/v2/memory/forget", json={"user_id": "amy", "target": target}
    )
    assert response.status_code == 422
    memory.forget.assert_not_awaited()


async def test_the_three_undo_bodies_reach_their_own_methods(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/forget/undo",
        json={"user_id": "amy", "forget_id": _FORGET_ID},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["applied"] is True and data["target"] == {"kind": "owner"}
    assert data["profile_items"] == 2
    assert memory.undo_forget.await_args.kwargs["forget_id"] == _FORGET_ID

    await client.post(
        "/api/v2/memory/forget/undo", json={"user_id": "amy", "image_id": _HASH}
    )
    memory.undo_forget_image.assert_awaited_once()
    await client.post(
        "/api/v2/memory/forget/undo",
        json={"group_id": "trip", "asker_id": "u_chen", "forget_id": _FORGET_ID},
    )
    memory.undo_group_forget.assert_awaited_once()


# ── delete is a forget ───────────────────────────────────────────────────


async def test_a_delete_answers_its_forget_ids(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post("/api/v2/memory/delete", json={"session_id": "s1"})
    assert response.status_code == 200, response.text
    assert response.json()["data"] == {
        "filters": ["session_id"],
        "count": 3,
        "forget_ids": [_FORGET_ID],
    }
    response = await client.post(
        "/api/v2/memory/delete_by_ids",
        json={"memory_type": "episode", "memory_ids": ["amy_ep_20261112_1"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["forget_ids"] == [_FORGET_ID]
