"""``POST /memory/forget`` and the two read-only picture routes.

What these pin:

- A forget is addressed by the picture's content hash and the owning user, and
  the route answers with what it touched — an empty 200 would let a caller
  believe a picture it never had was forgotten.
- The id is validated as a 64-hex hash *before* it reaches the store: it
  becomes a file name, so a free string there would be a path-traversal
  surface.
- ``GET /memory/images/{id}`` serves the original, and ``?variant=thumbnail``
  the decider-sized copy; an unknown id is a 404, never an empty 200.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.app import create_app
from everos.service.memory import ForgetResult, RevisionResult, UndoResult
from everos.service.memory._forget import ForgottenPicture
from tests.fakes import InMemoryImageStore, StubMemoryService, make_application

_IMAGE_ID = "a1" * 32


@pytest.fixture
def memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.forget_image.return_value = ForgetResult(
        image_id=_IMAGE_ID,
        image_memories=1,
        episodes=["ep_20261107_00000002"],
        profiles=0,
        pictures=[
            ForgottenPicture(
                entry_id="im_20261107_00000001",
                image_id=_IMAGE_ID,
                name="IMG_2231.jpg",
            )
        ],
    )
    return stub


@pytest.fixture
def pictures() -> InMemoryImageStore:
    """The picture store the application serves from, seeded by each test."""
    return InMemoryImageStore()


@pytest.fixture
async def client(
    memory: StubMemoryService, pictures: InMemoryImageStore
) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=memory, images=pictures))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_forget_returns_what_it_touched(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    response = await client.post(
        "/api/v2/memory/forget", json={"user_id": "me", "image_id": _IMAGE_ID}
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["image_memories"] == 1
    assert data["episodes"] == ["ep_20261107_00000002"]
    # The occurrence and the name it arrived under: what a caller needs to map
    # the memory back to its own input.
    assert data["pictures"] == [
        {
            "entry_id": "im_20261107_00000001",
            "image_id": _IMAGE_ID,
            "name": "IMG_2231.jpg",
        }
    ]
    memory.forget_image.assert_awaited()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"user_id": "me"},
        {"user_id": "me", "image_id": "not-a-hash"},
        {"user_id": "me", "image_id": "a1" * 31},  # one byte short
        {"user_id": "me", "image_id": "A1" * 32},  # uppercase is not the id
        {"user_id": "me", "image_id": "../../etc/passwd"},
        {"user_id": "a/b", "image_id": _IMAGE_ID},
        {"user_id": "me", "image_id": _IMAGE_ID, "unknown": 1},
    ],
)
async def test_forget_rejects_a_malformed_request(
    client: AsyncClient, memory: StubMemoryService, body: dict
) -> None:
    response = await client.post("/api/v2/memory/forget", json=body)

    assert response.status_code == 422
    memory.forget_image.assert_not_awaited()


async def test_original_is_served_with_its_media_type(
    client: AsyncClient, pictures: InMemoryImageStore
) -> None:
    pictures.images[_IMAGE_ID] = (b"\x89PNG-bytes", "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNG-bytes"


async def test_thumbnail_variant_serves_the_decider_copy(
    client: AsyncClient, pictures: InMemoryImageStore
) -> None:
    pictures.images[_IMAGE_ID] = (b"full", "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}?variant=thumbnail")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


async def test_an_unknown_picture_is_a_404(client: AsyncClient) -> None:
    response = await client.get("/api/v2/memory/images/" + "b2" * 32)

    assert response.status_code == 404


async def test_an_unknown_variant_is_rejected(
    client: AsyncClient, pictures: InMemoryImageStore
) -> None:
    pictures.images[_IMAGE_ID] = (b"full", "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}?variant=huge")

    assert response.status_code == 422


@pytest.fixture
def undo_memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.undo_forget_image.return_value = UndoResult(
        image_id=_IMAGE_ID,
        forget_id="fg_0123456789abcdef",
        applied=True,
        image_memories=1,
        episodes=("ep_20261107_00000004",),
        profiles=1,
        pictures=(
            ForgottenPicture(
                entry_id="im_20261107_00000001",
                image_id=_IMAGE_ID,
                name="IMG_2231.jpg",
            ),
        ),
    )
    return stub


@pytest.fixture
async def undo_client(undo_memory: StubMemoryService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=undo_memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_undo_reports_what_came_back(
    undo_client: AsyncClient, undo_memory: StubMemoryService
) -> None:
    response = await undo_client.post(
        "/api/v2/memory/forget/undo",
        json={"user_id": "me", "image_id": _IMAGE_ID},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["applied"] is True
    assert data["forget_id"] == "fg_0123456789abcdef"
    assert data["image_memories"] == 1
    assert data["episodes"] == ["ep_20261107_00000004"]
    assert data["profiles"] == 1
    assert data["pictures"] == [
        {
            "entry_id": "im_20261107_00000001",
            "image_id": _IMAGE_ID,
            "name": "IMG_2231.jpg",
        }
    ]
    undo_memory.undo_forget_image.assert_awaited_once()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"user_id": "me"},
        {"user_id": "me", "image_id": "not-a-hash"},
        {"user_id": "a/b", "image_id": _IMAGE_ID},
        {"user_id": "me", "image_id": _IMAGE_ID, "unknown": 1},
    ],
)
async def test_undo_rejects_a_malformed_request(
    undo_client: AsyncClient, undo_memory: StubMemoryService, body: dict
) -> None:
    response = await undo_client.post("/api/v2/memory/forget/undo", json=body)

    assert response.status_code == 422
    undo_memory.undo_forget_image.assert_not_awaited()


@pytest.fixture
def revise_memory() -> StubMemoryService:
    stub = StubMemoryService()
    stub.revise_image_memory.return_value = RevisionResult(
        image_id=_IMAGE_ID,
        entry_id="im_20261108_00000002",
        supersedes="im_20261107_00000001",
        description="a sign reading DOU SHE EST 2019",
        ocr="EST 2019",
    )
    return stub


@pytest.fixture
async def revise_client(revise_memory: StubMemoryService) -> AsyncIterator[AsyncClient]:
    app = create_app(application=make_application(memory=revise_memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_a_revision_answers_with_both_versions(
    revise_client: AsyncClient, revise_memory: StubMemoryService
) -> None:
    response = await revise_client.post(
        "/api/v2/memory/images/revise",
        json={"user_id": "me", "image_id": _IMAGE_ID, "description": "a sign"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["entry_id"] == "im_20261108_00000002"
    assert data["supersedes"] == "im_20261107_00000001"
    assert data["description"] == "a sign reading DOU SHE EST 2019"
    revise_memory.revise_image_memory.assert_awaited_once()


@pytest.mark.parametrize(
    "body",
    [
        {"user_id": "me", "image_id": _IMAGE_ID},  # nothing to change
        {"user_id": "me", "image_id": "not-a-hash", "description": "x"},
        {
            "user_id": "me",
            "image_id": _IMAGE_ID,
            "entry_id": "nope",
            "description": "x",
        },
        {"user_id": "me", "image_id": _IMAGE_ID, "description": "x", "extra": 1},
    ],
)
async def test_a_malformed_revision_is_refused(
    revise_client: AsyncClient, revise_memory: StubMemoryService, body: dict
) -> None:
    response = await revise_client.post("/api/v2/memory/images/revise", json=body)

    assert response.status_code == 422
    revise_memory.revise_image_memory.assert_not_awaited()
