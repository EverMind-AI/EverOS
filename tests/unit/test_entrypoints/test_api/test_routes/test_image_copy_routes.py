"""``GET /memory/images/{id}?variant=model``: the answering model's copy by id.

What these pin:

- ``model`` serves the ``[images] model_edge`` copy at ``cache_quality`` -- the
  same copy a search result names as ``model_path`` -- and ``thumbnail`` keeps
  serving the decider's size.
- A copy is labelled as what it is: PNG for a greyscale picture, else JPEG.
- The original stays the default, for a caller that wants the bytes as sent.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from tests.fakes import InMemoryImageStore, StubMemoryService, make_application

_IMAGE_ID = "c3" * 32


def _png(mode: str) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (8, 8), 128 if mode == "L" else (10, 120, 200)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class _SizedStore(InMemoryImageStore):
    """Serves a copy as the original's bytes and records the size it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[tuple[int, int]] = []

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        self.asked.append((edge, quality))
        return await super().read_derived(image_id, edge=edge, quality=quality)


@pytest.fixture
def pictures() -> _SizedStore:
    return _SizedStore()


@pytest.fixture
async def client(pictures: _SizedStore) -> AsyncIterator[AsyncClient]:
    app = create_app(
        application=make_application(memory=StubMemoryService(), images=pictures)
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_model_variant_serves_the_answer_copy(
    client: AsyncClient, pictures: _SizedStore
) -> None:
    pictures.images[_IMAGE_ID] = (b"\xff\xd8\xff-a-jpeg-copy", "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}?variant=model")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    images = load_settings().images
    assert pictures.asked == [(images.model_edge, images.cache_quality)]


async def test_a_greyscale_copy_is_labelled_png(
    client: AsyncClient, pictures: _SizedStore
) -> None:
    pictures.images[_IMAGE_ID] = (_png("L"), "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}?variant=model")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


async def test_thumbnail_keeps_the_decider_size(
    client: AsyncClient, pictures: _SizedStore
) -> None:
    pictures.images[_IMAGE_ID] = (_png("RGB"), "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}?variant=thumbnail")

    assert response.status_code == 200
    decider = load_settings().decider
    assert pictures.asked == [(decider.image_thumbnail_edge, decider.image_quality)]


async def test_the_original_stays_the_default(
    client: AsyncClient, pictures: _SizedStore
) -> None:
    pictures.images[_IMAGE_ID] = (_png("RGB"), "png")

    response = await client.get(f"/api/v2/memory/images/{_IMAGE_ID}")

    assert response.status_code == 200
    assert response.content == _png("RGB")
    assert pictures.asked == []


async def test_a_copy_of_an_unknown_picture_is_a_404(client: AsyncClient) -> None:
    response = await client.get("/api/v2/memory/images/" + "d4" * 32 + "?variant=model")

    assert response.status_code == 404
