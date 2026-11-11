"""What ingest refuses or cannot read is counted once and returned, never fatal.

``/add`` answered only ``message_count`` and ``status``: a picture over a limit
or a file the parser could not read vanished from the conversation without a
word to the caller. Ingest now counts each such item once -- ``rejected`` (a
picture refused: over a limit, not a picture, unreadable) or ``failed`` (a
non-text item the parser could not turn into text) -- and the counts travel to
the response.

Three defects of the same path are pinned here as well:

* a refused picture was handed to the parser anyway: a wasted multimodal call
  whose output was dropped, or -- with no ``[multimodal]`` configured -- an
  exception that failed the whole ``/add``;
* with ``[extraction] vision = false`` every picture was marked refused, then
  parsed, and the parsed text dropped, so no picture reached extraction;
* a parser call that failed (the real ``aparse_file`` raises
  ``LLMServiceError``) failed the whole ``/add`` instead of the one item.
"""

from __future__ import annotations

import base64
import io
from functools import partial
from typing import Any

import pytest
from PIL import Image

pytest.importorskip("everalgo.parser")

from everalgo.types import ParsedContent

from everos.component import parser as _parser_mod
from everos.component.images import ImageLimits
from everos.component.parser import build_raw_file
from everos.config import Settings
from everos.core.errors import LLMServiceError
from everos.memory.extract.ingest import process
from tests.fakes import InMemoryImageStore

_APARSE = "everos.component.parser.aparse_file"
_BUILD = partial(build_raw_file, settings_source=Settings)
_LLM = object()
"""Stands in for the multimodal client; the parser itself is faked."""


@pytest.fixture(autouse=True)
def _parser_module_loaded() -> None:
    assert _parser_mod is not None


def _png(colour: tuple[int, int, int] = (10, 120, 200), edge: int = 4) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (edge, edge), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _limits(**overrides: Any) -> ImageLimits:
    values: dict[str, Any] = {
        "max_bytes": 5 * 1024 * 1024,
        "max_per_message": 4,
        "max_request_bytes": 10 * 1024 * 1024,
        "max_per_request": 8,
        "formats": ("png", "jpeg", "webp", "gif"),
    }
    values.update(overrides)
    return ImageLimits(**values)


def _payload(*content: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": "s1",
        "messages": [
            {
                "sender_id": "u1",
                "role": "user",
                "timestamp": 1740564000000,
                "content": [{"type": "text", "text": "look at this"}, *content],
            }
        ],
    }


def _picture(data: bytes, name: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "image",
        "base64": base64.b64encode(data).decode(),
        "ext": "png",
    }
    if name:
        item["name"] = name
    return item


def _pdf() -> dict[str, Any]:
    return {
        "type": "pdf",
        "name": "menu.pdf",
        "base64": base64.b64encode(b"%PDF-1.4 fake").decode(),
        "ext": "pdf",
    }


class _Parser:
    """The fake parser: records what reached it, answers or fails."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.seen: list[str] = []

    async def __call__(self, raw: Any, *, llm: Any) -> ParsedContent:
        self.seen.append(raw.extension)
        if llm is None:
            from everos.component.llm import LLMNotConfiguredError

            raise LLMNotConfiguredError("no [multimodal]")
        if self.error is not None:
            raise self.error
        return ParsedContent(text="PARSED TEXT")


async def test_a_refused_picture_is_counted_and_never_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _Parser()
    monkeypatch.setattr(_APARSE, parser)

    result = await process(
        _payload(_picture(_png(edge=64))),
        multimodal_llm=None,  # nothing configured: parsing it would raise
        raw_file_builder=_BUILD,
        image_store=InMemoryImageStore(),
        image_limits=_limits(max_bytes=10),
    )

    assert (result.rejected_count, result.failed_count) == (1, 0)
    assert result.unparsed_non_text_count == 1  # once, not twice
    assert parser.seen == []
    assert result.messages[0].text == "look at this"


async def test_a_stored_picture_is_neither_rejected_nor_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _Parser()
    monkeypatch.setattr(_APARSE, parser)

    result = await process(
        _payload(_picture(_png())),
        multimodal_llm=_LLM,
        raw_file_builder=_BUILD,
        image_store=InMemoryImageStore(),
        image_limits=_limits(),
    )

    assert (result.rejected_count, result.failed_count) == (0, 0)
    assert parser.seen == []
    assert "[IMAGE: " in result.messages[0].text


async def test_with_vision_off_a_parsed_picture_reaches_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _Parser()
    monkeypatch.setattr(_APARSE, parser)

    result = await process(
        _payload(_picture(_png(), name="receipt.png")),
        multimodal_llm=_LLM,
        raw_file_builder=_BUILD,
        image_store=InMemoryImageStore(),
        image_limits=_limits(),
        vision=False,
    )

    assert parser.seen == ["png"]
    assert result.messages[0].text == "look at this\n[IMAGE: receipt.png]\nPARSED TEXT"
    assert (result.rejected_count, result.failed_count) == (0, 0)


async def test_a_parser_failure_fails_the_item_not_the_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the real ``aparse_file`` raises when its model call fails."""
    parser = _Parser(error=LLMServiceError("multimodal endpoint timed out"))
    monkeypatch.setattr(_APARSE, parser)

    result = await process(
        _payload(_pdf()),
        multimodal_llm=_LLM,
        raw_file_builder=_BUILD,
        image_store=InMemoryImageStore(),
        image_limits=_limits(),
    )

    assert parser.seen == ["pdf"]
    assert (result.rejected_count, result.failed_count) == (0, 1)
    assert result.messages[0].text == "look at this"


async def test_counts_add_up_across_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_APARSE, _Parser(error=LLMServiceError("down")))
    payload = _payload(_picture(_png(edge=64)), _pdf())
    payload["messages"].append(
        {
            "sender_id": "u1",
            "role": "user",
            "timestamp": 1740564001000,
            "content": [_picture(b"not a picture")],
        }
    )

    result = await process(
        payload,
        multimodal_llm=_LLM,
        raw_file_builder=_BUILD,
        image_store=InMemoryImageStore(),
        image_limits=_limits(max_bytes=100),
    )

    assert (result.rejected_count, result.failed_count) == (2, 1)
    assert result.unparsed_non_text_count == 3
