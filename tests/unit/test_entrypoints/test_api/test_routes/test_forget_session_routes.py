"""``POST /memory/forget/session`` and its undo.

What these pin:

- A conversation's forget reports what it marked and which picture *files* it
  took out of circulation, so a caller can see the difference between "the
  memory is hidden" and "the picture was only here".
- The undo repeats the scope fields and answers ``applied`` — ``false`` is a
  real answer for a conversation that was never forgotten, not an error.
- Both are validated like the picture forget: path-safe user id, a bounded
  reason, no unknown fields.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.app import create_app
from everos.service import ForgetResult, ForgetTarget, UndoResult
from tests.fakes import StubMemoryService, make_application

_IMAGE_ID = "a1" * 32


@pytest.fixture
def memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.forget_session.return_value = ForgetResult(
        forget_id="fg_0123456789abcdef",
        dry_run=False,
        target=ForgetTarget(kind="session", session_id="s1"),
        episodes=("me_ep_20261107_00000001",),
        image_memories=("me_im_20261107_00000001",),
        images=(_IMAGE_ID,),
    )
    stub.undo_forget_session.return_value = UndoResult(
        forget_id="fg_0123456789abcdef",
        applied=True,
        episodes=("me_ep_20261107_00000001",),
        image_memories=("me_im_20261107_00000001",),
        images=(_IMAGE_ID,),
    )
    return stub


@pytest.fixture
async def client(memory: StubMemoryService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_forget_session_reports_what_it_marked(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/forget/session", json={"user_id": "me", "session_id": "s1"}
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["episodes"] == ["ep_20261107_00000001"]
    assert data["image_memories"] == 1
    assert data["images"] == [_IMAGE_ID]
    assert data["forget_id"] == "fg_0123456789abcdef"
    memory.forget_session.assert_awaited_once()


async def test_undo_session_reports_what_came_back(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/forget/session/undo",
        json={"user_id": "me", "session_id": "s1"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["applied"] is True
    assert data["episodes"] == ["ep_20261107_00000001"]
    assert data["images"] == [_IMAGE_ID]
    memory.undo_forget_session.assert_awaited_once()


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/v2/memory/forget/session", {"user_id": "me"}),
        ("/api/v2/memory/forget/session", {"session_id": "s1"}),
        ("/api/v2/memory/forget/session", {"user_id": "a/b", "session_id": "s1"}),
        (
            "/api/v2/memory/forget/session",
            {"user_id": "me", "session_id": "s1", "unknown": 1},
        ),
        ("/api/v2/memory/forget/session/undo", {"user_id": "me"}),
    ],
)
async def test_malformed_session_requests_are_refused(
    client: AsyncClient, memory: StubMemoryService, path: str, body: dict
) -> None:
    response = await client.post(path, json=body)

    assert response.status_code == 422
    memory.forget_session.assert_not_awaited()
    memory.undo_forget_session.assert_not_awaited()
