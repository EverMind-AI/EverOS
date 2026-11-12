"""The group bodies of the forget, undo and delete routes (section 5.5).

docs/group-memory.md, sections 5.5 and 15. Protocol translation only; the
service is a stub. What these pin:

- a body naming ``group_id`` is a group forget, a ``user_id`` with
  ``image_id`` the personal picture forget it always was; each reaches its own
  facade method; a group forget is a dry run unless ``dry_run`` is false;
- each of the five targets reaches the service as it was sent, with the
  gateway's identity header passed through for the service to compare;
- a group forget's answer names what it touched, and its ``forget_id`` is what
  the undo takes;
- malformed group bodies are 422 before the service is called: an unknown
  target kind, a target missing its fields, a picture id that is no hash, a
  ``forget_id`` of the wrong shape, personal fields on a group body;
- a group-scoped delete is the service's group delete and answers with the
  forget's id, while a personal delete's answer keeps its exact shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.service import GroupForgetResult, GroupForgetTarget, GroupUndoResult
from everos.service.memory import DeleteResult, ForgetResult
from tests.fakes import StubMemoryService, make_application

_HASH = "ab" * 32


@pytest.fixture
def memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.forget_in_group.side_effect = lambda **kw: GroupForgetResult(
        forget_id="" if kw["dry_run"] else "fg_0123456789abcdef",
        dry_run=kw["dry_run"],
        target=kw["target"],
        messages=(("s1", "m-1"),),
        episodes=("trip_gep_20261111_00000001",),
        rewritten=1,
        new_episodes=() if kw["dry_run"] else ("trip_gep_20261111_00000002",),
    )
    stub.undo_group_forget.return_value = GroupUndoResult(
        forget_id="fg_0123456789abcdef",
        applied=True,
        episodes=("trip_gep_20261111_00000001",),
        withdrawn=("trip_gep_20261111_00000002",),
    )
    return stub


@pytest.fixture
async def client(memory: StubMemoryService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


def _body(target: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"group_id": "trip", "asker_id": "u_betty", "target": target, **extra}


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            {"kind": "message", "session_id": "s1", "message_ids": ["m-1", "m-2"]},
            GroupForgetTarget(
                kind="message", session_id="s1", message_ids=("m-1", "m-2")
            ),
        ),
        (
            {"kind": "member", "member_id": "u_betty"},
            GroupForgetTarget(kind="member", member_id="u_betty"),
        ),
        (
            {"kind": "session", "session_id": "s1"},
            GroupForgetTarget(kind="session", session_id="s1"),
        ),
        (
            {"kind": "record", "record_ids": ["trip_gep_20261111_00000001"]},
            GroupForgetTarget(
                kind="record", record_ids=("trip_gep_20261111_00000001",)
            ),
        ),
        (
            {"kind": "image", "image_id": _HASH},
            GroupForgetTarget(kind="image", image_id=_HASH),
        ),
    ],
)
async def test_every_target_reaches_the_group_forget(
    client: AsyncClient,
    memory: StubMemoryService,
    target: dict[str, Any],
    expected: GroupForgetTarget,
) -> None:
    response = await client.post(
        "/api/v2/memory/forget",
        json=_body(target, reason="sent by mistake", dry_run=False),
    )

    assert response.status_code == 200, response.text
    call = memory.forget_in_group.await_args.kwargs
    assert call["target"] == expected
    assert (call["group_id"], call["asker_id"], call["reason"]) == (
        "trip",
        "u_betty",
        "sent by mistake",
    )
    assert call["dry_run"] is False
    memory.forget_image.assert_not_awaited()
    data = response.json()["data"]
    assert data["forget_id"] == "fg_0123456789abcdef"
    assert data["messages"] == [{"session_id": "s1", "message_id": "m-1"}]
    assert data["new_episodes"] == ["trip_gep_20261111_00000002"]
    assert data["target"]["kind"] == expected.kind


async def test_a_dry_run_says_so_and_is_the_default(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/forget",
        json=_body({"kind": "member", "member_id": "u_betty"}),
    )
    assert memory.forget_in_group.await_args.kwargs["dry_run"] is True
    data = response.json()["data"]
    assert (data["dry_run"], data["forget_id"], data["new_episodes"]) == (
        True,
        "",
        [],
    )
    assert data["rewritten"] == 1


async def test_a_personal_body_is_still_the_picture_forget(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    memory.forget_image.return_value = ForgetResult(
        image_id=_HASH, image_memories=0, episodes=[], profiles=0
    )
    response = await client.post(
        "/api/v2/memory/forget", json={"user_id": "me", "image_id": _HASH}
    )
    assert response.status_code == 200
    memory.forget_in_group.assert_not_awaited()
    assert response.json()["data"]["image_id"] == _HASH


async def test_the_identity_header_reaches_the_service(
    client: AsyncClient, memory: StubMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_ACCESS__IDENTITY_HEADER", "X-EverOS-Asker")
    load_settings.cache_clear()
    await client.post(
        "/api/v2/memory/forget",
        json=_body({"kind": "member", "member_id": "u_betty"}),
        headers={"X-EverOS-Asker": "u_dan"},
    )
    assert memory.forget_in_group.await_args.kwargs["asserted_asker"] == "u_dan"


@pytest.mark.parametrize(
    "body",
    [
        _body({"kind": "everything"}),
        _body({"kind": "message", "session_id": "s1"}),
        _body({"kind": "message", "session_id": "s1", "message_ids": []}),
        _body({"kind": "image", "image_id": "../../etc/passwd"}),
        _body({"kind": "record", "record_ids": ["a/b"]}),
        {**_body({"kind": "member", "member_id": "u"}), "user_id": "u"},
        {"group_id": "trip", "asker_id": "u"},
        {
            "group_id": "../x",
            "asker_id": "u",
            "target": {"kind": "member", "member_id": "u"},
        },
    ],
)
async def test_malformed_group_bodies_are_refused(
    client: AsyncClient, memory: StubMemoryService, body: dict[str, Any]
) -> None:
    response = await client.post("/api/v2/memory/forget", json=body)
    assert response.status_code == 422, response.text
    memory.forget_in_group.assert_not_awaited()
    memory.forget_image.assert_not_awaited()


async def test_an_undo_takes_the_forget_id(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/forget/undo",
        json={"group_id": "trip", "asker_id": "u_betty", "forget_id": "fg_" + "a" * 16},
    )
    assert response.status_code == 200, response.text
    call = memory.undo_group_forget.await_args.kwargs
    assert (call["group_id"], call["forget_id"]) == ("trip", "fg_" + "a" * 16)
    data = response.json()["data"]
    assert data["applied"] is True
    assert data["withdrawn"] == ["trip_gep_20261111_00000002"]
    bad = await client.post(
        "/api/v2/memory/forget/undo",
        json={"group_id": "trip", "asker_id": "u_betty", "forget_id": "x"},
    )
    assert bad.status_code == 422
    memory.undo_forget_image.assert_not_awaited()


async def test_a_group_delete_answers_with_its_forget(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    memory.delete_in_group.return_value = (
        DeleteResult(filters=["group_id", "session_id"], count=3),
        GroupForgetResult(
            forget_id="fg_0123456789abcdef",
            dry_run=False,
            target=GroupForgetTarget(kind="session", session_id="s1"),
        ),
    )
    response = await client.post(
        "/api/v2/memory/delete",
        json={"group_id": "trip", "asker_id": "u_chen", "session_id": "s1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"] == {
        "filters": ["group_id", "session_id"],
        "count": 3,
        "forget_id": "fg_0123456789abcdef",
    }
    call = memory.delete_in_group.await_args.kwargs
    assert (call["session_id"], call["member_id"]) == ("s1", None)
    memory.delete.assert_not_awaited()

    memory.delete.return_value = DeleteResult(
        filters=["user_id"], count=2, forget_ids=("fg_0123456789abcdee",)
    )
    personal = await client.post("/api/v2/memory/delete", json={"user_id": "u_chen"})
    assert personal.json()["data"] == {
        "filters": ["user_id"],
        "count": 2,
        "forget_ids": ["fg_0123456789abcdee"],
    }


@pytest.mark.parametrize(
    "body",
    [
        {"group_id": "trip", "agent_id": "a", "asker_id": "u"},
        {"group_id": "trip", "user_id": "u", "session_id": "s", "asker_id": "u"},
        {"user_id": "u", "asker_id": "u"},
    ],
)
async def test_malformed_group_deletes_are_refused(
    client: AsyncClient, memory: StubMemoryService, body: dict[str, Any]
) -> None:
    response = await client.post("/api/v2/memory/delete", json=body)
    assert response.status_code == 422, response.text
    memory.delete_in_group.assert_not_awaited()
