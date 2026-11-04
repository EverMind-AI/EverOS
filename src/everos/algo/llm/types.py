"""Request types for ``LLMClient.achat``; they serialise to the OpenAI Chat Completions message shape."""

from __future__ import annotations

__all__ = [
    "ContentPart",
    "ImageUrlInner",
    "ImageUrlPart",
    "LLMMessage",
    "TextPart",
    "image_url_part_from_bytes",
]

import base64
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    """Text content part, serialised as ``{"type": "text", "text": ...}``."""

    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(extra="ignore")


class ImageUrlInner(BaseModel):
    """Inner object of an ``image_url`` content part."""

    url: str
    detail: Literal["low", "high", "auto"] | None = None

    model_config = ConfigDict(extra="ignore")


class ImageUrlPart(BaseModel):
    """File content part, serialised as ``{"type": "image_url", "image_url": {...}}``.

    Despite the wire name, the URL may be a data URI of any MIME type the upstream model accepts, such as
    ``image/png``, ``application/pdf`` or ``audio/wav``.
    """

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlInner

    model_config = ConfigDict(extra="ignore")


ContentPart = Annotated[TextPart | ImageUrlPart, Field(discriminator="type")]
"""A content part of a multimodal message, discriminated by ``type``."""


def image_url_part_from_bytes(data: bytes, mime: str) -> ImageUrlPart:
    """Wrap raw file bytes in an ``ImageUrlPart`` as a base64 data URI.

    Args:
        data: File payload, such as image, PDF or audio bytes.
        mime: MIME type of ``data``, e.g. ``image/png`` or ``application/pdf``.

    Returns:
        A part whose URL is ``data:<mime>;base64,<encoded data>``.
    """
    b64 = base64.b64encode(data).decode("ascii")
    return ImageUrlPart(image_url=ImageUrlInner(url=f"data:{mime};base64,{b64}"))


class LLMMessage(BaseModel):
    """One message of a chat request.

    Distinct from the domain record ``everos.algo.types.chat.ChatMessage``: an operator renders domain messages into
    text and sends that text in an ``LLMMessage``, never converting one type into the other.

    Attributes:
        role: Who speaks: ``system``, ``user`` or ``assistant``.
        content: Plain text, or a list of content parts for multimodal input.
    """

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    model_config = ConfigDict(extra="ignore")
