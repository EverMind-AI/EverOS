"""Multimodal content normalisation.

Coerces raw message ``content`` into a list of ContentItem dicts and derives the
flat ``text`` the boundary detector, the md writer and every text-only model
consume.

Two kinds of non-text item are handled differently, and the difference is the
whole design:

- **Pictures** the extraction model can read are *stored*, not transcribed.
  The item gains ``image_id`` (the SHA-256 of its bytes), ``original_path``,
  ``media_type``, ``width`` and ``height``, and renders as a reference
  ``[IMAGE: <image_id>]`` in the text. The model is handed the picture itself
  alongside that text (see :mod:`everos.memory.extract.pipeline.user_memory`),
  so a description written from it can name what only the picture shows — the
  sign, the receipt, the price board — instead of whatever a lossy caption
  round-trip left behind.
- **Everything else** (audio, pdf, doc, html, email) contributes its
  ``parsed_content`` (populated by the parser hook,
  ``memory/extract/parser``) rendered as a provenance tag
  ``[TYPE: name]\\n{parsed_content}``; items still lacking ``parsed_content``
  are counted so callers can surface them via
  ``IngestResult.unparsed_non_text_count``.

A picture that no format accepts, or that breaches a configured limit, is
counted and dropped rather than failing the request: the text around it is
still worth remembering, and one corrupt upload must not cost a conversation.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from everos.component.images import ImageLimits, StoredImage, dedupe, hash_bytes
from everos.core.observability.logging import get_logger
from everos.memory.ports.images import ImageStore

logger = get_logger(__name__)

IMAGE_TYPE = "image"
"""The ``ContentItem.type`` value a picture arrives under."""


@dataclass
class PictureBudget:
    """What one add request has already spent on pictures.

    Two numbers, because they answer two different questions: ``bytes_used``
    is the request's picture payload (what the extraction model is billed for)
    and ``seen`` is which pictures have already been counted, so forwarding the
    same picture twice is not charged twice. One instance per request; every
    message of that request writes into it.

    Attributes:
        max_bytes: The request's decoded picture byte ceiling.
        max_pictures: The request's picture-count ceiling.
        bytes_used: Bytes charged so far.
        pictures: Distinct pictures charged so far.
        seen: Content hashes already charged.
    """

    max_bytes: int
    max_pictures: int
    bytes_used: int = 0
    pictures: int = 0
    seen: set[str] = field(default_factory=set)

    def admits(self, data: bytes) -> bool:
        """Whether one more picture of ``data`` fits the request."""
        digest = hash_bytes(data)
        if digest in self.seen:
            return True
        return (
            self.bytes_used + len(data) <= self.max_bytes
            and self.pictures + 1 <= self.max_pictures
        )

    def charge(self, data: bytes) -> None:
        """Count one picture against the request."""
        digest = hash_bytes(data)
        if digest in self.seen:
            return
        self.seen.add(digest)
        self.bytes_used += len(data)
        self.pictures += 1


def coerce_items(
    content: str | list[dict[str, Any]] | list[Any],
) -> list[dict[str, Any]]:
    """Coerce ``content`` into a list of ContentItem dicts.

    Accepts the simplified ``str`` form (DTO sugar) or a list of
    ContentItem-shaped dicts / Pydantic models.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_coerce_item(item) for item in content]


def derive_text(items: list[dict[str, Any]]) -> tuple[str, int]:
    """Render items into the derived ``text`` + count still-unparsed non-text.

    Returns ``(text, non_text_count)``: ``text`` is the newline-joined render
    of all items; ``non_text_count`` is the number of non-text items that
    contribute nothing — a picture is not one of them once stored, because its
    reference is what the extractor reads.
    """
    parts: list[str] = []
    non_text = 0
    for item in items:
        rendered = _render_item(item)
        if rendered:
            parts.append(rendered)
        elif item.get("type") != "text":
            non_text += 1
            logger.warning(
                "multimodal_content_not_parsed",
                content_type=item.get("type", "unknown"),
            )
    return "\n".join(parts), non_text


def normalise_content(
    content: str | list[dict[str, Any]] | list[Any],
) -> tuple[list[dict[str, Any]], str, int]:
    """Coerce + derive in one step (text-only path; no parsing).

    Returns ``(content_items, text, non_text_count)``. The ingest service
    splits this into :func:`coerce_items` / :func:`derive_text` so the picture
    store and the parser hook can run between them; this wrapper is kept for
    text-only callers.
    """
    items = coerce_items(content)
    text, non_text = derive_text(items)
    return items, text, non_text


async def store_images(
    items: list[dict[str, Any]],
    *,
    store: ImageStore | None,
    limits: ImageLimits,
    raw_file_builder: Any,
    budget: PictureBudget,
) -> list[StoredImage]:
    """Store every picture among ``items``; annotate each item in place.

    ``budget`` spans the whole request rather than one message: a request that
    carries a hundred small pictures costs the extraction model a hundred image
    tokens each, so the bound belongs at the request. It also remembers which
    pictures were charged, because forwarded bytes cost the model nothing the
    second time.

    A picture is stored once per content hash. The second arrival of the same
    bytes adds a reference and no file, and both items carry the same
    ``image_id`` — which is precisely what makes "forget this picture" a
    single call rather than one per occurrence.

    Returns the distinct pictures in first-seen order, so a caller can hand
    them to a model without sending the same bytes twice.
    """
    if store is None:
        for item in items:
            if item.get("type") == IMAGE_TYPE:
                item["status"] = "rejected"
                item["reason"] = "picture storage is not configured"
        return []

    stored: list[StoredImage] = []
    per_message = 0
    for item in items:
        if item.get("type") != IMAGE_TYPE:
            continue
        per_message += 1
        if per_message > limits.max_per_message:
            _reject(item, "over the per-message picture limit", logger, limits)
            continue
        try:
            data = await _bytes_of(item, raw_file_builder=raw_file_builder)
        except ValueError as exc:
            _reject(item, str(exc), logger, limits)
            continue
        if not budget.admits(data):
            _reject(item, "over the per-request picture limit", logger, limits)
            continue
        try:
            image = await store.store(data, limits=limits)
        except ValueError as exc:
            _reject(item, str(exc), logger, limits)
            continue
        budget.charge(data)
        item["image_id"] = image.image_id
        item["media_type"] = image.media_type
        item["width"] = image.width
        item["height"] = image.height
        item["status"] = "stored"
        stored.append(image)

    return dedupe(stored)


async def _bytes_of(item: dict[str, Any], *, raw_file_builder: Any) -> bytes:
    """The picture's bytes, from whichever form the content item carries.

    ``base64`` is decoded directly; ``uri`` (including ``file://``) goes through
    the same hydrator the parser uses, so a local path is subject to the
    ``file_uri_allow_dirs`` guardrail rather than being read blindly.
    """
    encoded = item.get("base64")
    if encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("picture base64 is not valid base64") from exc
    uri = item.get("uri")
    if not uri:
        raise ValueError("picture content item carries neither uri nor base64")
    raw = await raw_file_builder(item)
    content = getattr(raw, "content", None)
    if not content:
        raise ValueError(f"picture uri {uri!r} could not be read")
    return bytes(content)


def _reject(item: dict[str, Any], reason: str, log: Any, limits: ImageLimits) -> None:
    """Mark one picture unusable and count it; never fails the message."""
    item["status"] = "rejected"
    item["reason"] = reason
    log.warning(
        "memory.images.rejected",
        reason=reason,
        max_bytes=limits.max_bytes,
        max_per_message=limits.max_per_message,
    )


def image_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stored pictures among ``items``, in order."""
    return [
        item
        for item in items
        if item.get("type") == IMAGE_TYPE and item.get("status") == "stored"
    ]


def _render_item(item: dict[str, Any]) -> str | None:
    """Render one item to text, or ``None`` if it contributes nothing.

    Text items yield their ``text``. A stored picture yields a reference, never
    a description: the description is a *memory* of the picture, written
    later, and putting one here would let a stale caption stand in for the
    picture in every downstream extraction. Other non-text items yield
    ``[TYPE: name]\\n{parsed_content}`` once parsed.
    """
    kind = item.get("type")
    if kind == "text":
        text = item.get("text")
        return str(text) if text else None
    if kind == IMAGE_TYPE:
        image_id = item.get("image_id")
        return f"[IMAGE: {image_id}]" if image_id else None
    parsed = item.get("parsed_content")
    if not parsed:
        return None
    tag_kind = str(kind or "file").upper()
    name = item.get("name") or ""
    tag = f"[{tag_kind}: {name}]" if name else f"[{tag_kind}]"
    return f"{tag}\n{parsed}"


def _coerce_item(item: Any) -> dict[str, Any]:
    """Coerce ``item`` (dict or pydantic model) into a plain dict."""
    if hasattr(item, "model_dump"):
        return dict(item.model_dump())
    if isinstance(item, dict):
        return dict(item)
    return {"type": "unknown", "raw": repr(item)}
