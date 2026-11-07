"""Ingest content coercion, picture storage and text derivation.

What these pin:

- A picture is **stored and referenced by its content hash**, never rendered
  as a description: ``[IMAGE: <sha256>]`` is what the extraction model sees in
  the conversation text, and the bytes travel beside it as an image part. A
  description written at ingest time would stand in for the picture in every
  downstream extraction.
- Everything else non-text still renders as ``[TYPE: name]\\n{content}`` once
  parsed, so the parser route keeps working unchanged.
- A picture no format accepts is counted and dropped; it never fails the
  message it arrived in.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from everos.component.images import ImageLimits
from everos.memory.extract.ingest.multimodal import (
    PictureBudget,
    coerce_items,
    derive_text,
    image_items,
    normalise_content,
    store_images,
)
from everos.memory.ports.images import ImageStore
from tests.fakes import InMemoryImageStore


def _limits(**overrides: object) -> ImageLimits:
    values: dict[str, object] = {
        "max_bytes": 5 * 1024 * 1024,
        "max_per_message": 4,
        "max_request_bytes": 10 * 1024 * 1024,
        "max_per_request": 8,
        "formats": ("png", "jpeg", "webp", "gif"),
    }
    values.update(overrides)
    return ImageLimits(**values)  # type: ignore[arg-type]


def _png_bytes(colour: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    """A tiny valid PNG, distinct per colour so hashes differ."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _budget(*, max_bytes: int | None = None, max_pictures: int = 8) -> PictureBudget:
    return PictureBudget(
        max_bytes=max_bytes if max_bytes is not None else 10 * 1024 * 1024,
        max_pictures=max_pictures,
    )


def _image_item(data: bytes) -> dict[str, object]:
    return {
        "type": "image",
        "base64": base64.b64encode(data).decode(),
        "ext": "png",
    }


def test_coerce_str_to_text_item() -> None:
    assert coerce_items("hi") == [{"type": "text", "text": "hi"}]


def test_derive_text_renders_parsed_nontext_as_tag() -> None:
    items = [
        {"type": "text", "text": "before"},
        {"type": "pdf", "name": "p.pdf", "parsed_content": "PAGE TEXT"},
        {"type": "text", "text": "after"},
    ]
    text, non_text = derive_text(items)

    assert "[PDF: p.pdf]\nPAGE TEXT" in text
    assert text.startswith("before")
    assert text.endswith("after")
    assert non_text == 0


def test_derive_text_counts_unparsed_nontext() -> None:
    text, non_text = derive_text([{"type": "pdf", "uri": "x"}])
    assert text == ""
    assert non_text == 1


def test_derive_text_tag_without_name() -> None:
    text, _ = derive_text([{"type": "pdf", "parsed_content": "DOC"}])
    assert text == "[PDF]\nDOC"


def test_normalise_content_text_only_unchanged() -> None:
    items, text, non_text = normalise_content("hello")
    assert items == [{"type": "text", "text": "hello"}]
    assert text == "hello"
    assert non_text == 0


async def test_stored_picture_renders_as_a_reference() -> None:
    """The id is what the text carries; the description is written later."""
    store = InMemoryImageStore()
    items = [{"type": "text", "text": "look"}, _image_item(_png_bytes())]

    stored = await store_images(
        items,
        store=store,
        limits=_limits(),
        raw_file_builder=_unused_builder,
        budget=_budget(),
    )

    assert len(stored) == 1
    assert items[1]["image_id"] == stored[0].image_id
    text, non_text = derive_text(items)
    assert text == f"look\n[IMAGE: {stored[0].image_id}]"
    assert non_text == 0


async def test_the_same_picture_twice_is_one_file_two_references() -> None:
    """Deduplication is by content: the budget counts the bytes once."""
    store = InMemoryImageStore()
    data = _png_bytes()
    items = [_image_item(data), _image_item(data)]
    budget = _budget()

    stored = await store_images(
        items,
        store=store,
        limits=_limits(),
        raw_file_builder=_unused_builder,
        budget=budget,
    )

    assert len(stored) == 1
    assert len(store.images) == 1
    assert store.refs[stored[0].image_id] == 2
    assert budget.bytes_used == len(data)
    assert budget.pictures == 1
    assert items[0]["image_id"] == items[1]["image_id"]


async def test_a_request_over_its_byte_budget_drops_later_pictures() -> None:
    store = InMemoryImageStore()
    data = _png_bytes()
    items = [_image_item(data), _image_item(_png_bytes((1, 2, 3)))]

    stored = await store_images(
        items,
        store=store,
        limits=_limits(max_request_bytes=len(data) + 1),
        raw_file_builder=_unused_builder,
        budget=_budget(max_bytes=len(data) + 1),
    )

    assert len(stored) == 1
    assert items[0]["status"] == "stored"
    assert items[1]["status"] == "rejected"
    assert "per-request" in str(items[1]["reason"])


async def test_a_message_over_its_picture_count_keeps_the_first_ones() -> None:
    store = InMemoryImageStore()
    items = [
        _image_item(_png_bytes((1, 1, 1))),
        _image_item(_png_bytes((2, 2, 2))),
        _image_item(_png_bytes((3, 3, 3))),
    ]

    stored = await store_images(
        items,
        store=store,
        limits=_limits(max_per_message=2),
        raw_file_builder=_unused_builder,
        budget=_budget(),
    )

    assert len(stored) == 2
    assert items[2]["status"] == "rejected"
    assert len(image_items(items)) == 2


async def test_a_non_picture_is_rejected_not_raised() -> None:
    """One corrupt upload must not cost the conversation it arrived in."""
    store = InMemoryImageStore()
    items = [{"type": "image", "base64": base64.b64encode(b"not an image").decode()}]

    stored = await store_images(
        items,
        store=store,
        limits=_limits(),
        raw_file_builder=_unused_builder,
        budget=_budget(),
    )

    assert stored == []
    assert items[0]["status"] == "rejected"
    text, non_text = derive_text(items)
    assert text == ""
    assert non_text == 1


async def test_without_a_store_every_picture_is_reported() -> None:
    items = [_image_item(_png_bytes())]

    stored = await store_images(
        items,
        store=None,
        limits=_limits(),
        raw_file_builder=_unused_builder,
        budget=_budget(),
    )

    assert stored == []
    assert items[0]["status"] == "rejected"


async def _unused_builder(_item: dict[str, object]) -> object:
    raise AssertionError("base64 items need no host hydration")


def _assert_is_port() -> None:
    """The fake satisfies the port, so a test cannot drift from the contract."""
    store: ImageStore = InMemoryImageStore()
    assert store is not None
