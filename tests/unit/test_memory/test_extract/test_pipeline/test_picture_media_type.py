"""A picture's copy is labelled as what it is (A-5).

The extraction model and the decider are handed the model-size copy of a
picture; a greyscale picture's copy is PNG and every other is JPEG
(:func:`~everos.component.images.scaled_copy`). The label sent with the bytes
was ``image/jpeg`` for both, which a strict endpoint may refuse.
"""

from __future__ import annotations

import io

from everalgo.types import ChatMessage, MemCell
from PIL import Image

from everos.component.images import ImageLimits, copy_media_type, scaled_copy
from everos.memory.extract.pipeline.pictures import pictures_of
from tests.fakes.stores import InMemoryImageStore

_LIMITS = ImageLimits(
    max_bytes=5 * 1024 * 1024,
    max_per_message=4,
    max_request_bytes=10 * 1024 * 1024,
    max_per_request=8,
    formats=("png", "jpeg"),
    model_edge=1024,
    cache_quality=85,
)


def _encoded(mode: str, colour: object, fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (40, 30), colour).save(buffer, format=fmt)
    return buffer.getvalue()


def test_a_copy_is_labelled_by_its_bytes() -> None:
    grey, _ = scaled_copy(_encoded("L", 128, "PNG"), edge=32, quality=85)
    colour, _ = scaled_copy(_encoded("RGB", (200, 10, 10), "PNG"), edge=32, quality=85)
    assert copy_media_type(grey) == "image/png"
    assert copy_media_type(colour) == "image/jpeg"


async def test_the_extraction_copy_of_a_greyscale_picture_is_png() -> None:
    store = InMemoryImageStore()
    grey = await store.store(_encoded("L", 128, "PNG"), limits=_LIMITS)
    photo = await store.store(_encoded("RGB", (10, 200, 10), "JPEG"), limits=_LIMITS)
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content=f"[IMAGE: {grey.image_id}] and [IMAGE: {photo.image_id}]",
                sender_id="amy",
                timestamp=1_700_000_000_000,
            )
        ],
        timestamp=1_700_000_000_000,
    )
    pictures = await pictures_of(cell, images=store, limits=_LIMITS)
    assert [p.mime for p in pictures] == ["image/png", "image/jpeg"]
