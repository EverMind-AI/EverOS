"""Tests for enrich_content_items (component.parser.aparse_file is monkeypatched)."""

from __future__ import annotations

import base64
from typing import Any

import pytest

pytest.importorskip("everalgo.parser")

from everalgo.llm import LLMError
from everalgo.types import ParsedContent

from everos.component import parser as _parser_mod
from everos.core.errors import UnsupportedModalityError
from everos.memory.extract.parser import enrich_content_items

_APARSE_FILE_TARGET = "everos.component.parser.aparse_file"


@pytest.fixture(autouse=True)
def _ensure_parser_module_imported() -> None:
    """Force ``everos.component.parser`` into sys.modules before monkeypatch.

    ``enrich_content_items`` does ``from everos.component.parser import
    aparse_file`` inside its body. If the module hasn't been imported yet
    when monkeypatch runs, the ``from`` import creates a fresh binding
    to the real function, bypassing the patch.
    """
    assert _parser_mod is not None


def _img_item() -> dict[str, Any]:
    return {
        "type": "image",
        "base64": base64.b64encode(b"\x89PNG").decode(),
        "ext": "png",
    }


def _html_b64_item() -> dict[str, Any]:
    return {
        "type": "html",
        "base64": base64.b64encode(b"<html><body>v9.9.9</body></html>").decode(),
        "ext": "html",
    }


def _html_uri_item() -> dict[str, Any]:
    return {"type": "html", "uri": "https://example.com/page.html"}


async def test_enrich_backfills_parsed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aparse(raw_file: Any) -> ParsedContent:
        return ParsedContent(text="OCR RESULT")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    items: list[dict[str, Any]] = [{"type": "text", "text": "hi"}, _img_item()]
    await enrich_content_items(items, max_concurrency=2)

    assert items[1]["parsed_content"] == "OCR RESULT"
    assert items[1]["parse_status"] == "success"
    assert "parsed_content" not in items[0]


async def test_enrich_unsupported_modality_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aparse(raw_file: Any) -> ParsedContent:
        raise UnsupportedModalityError("video deferred")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    with pytest.raises(UnsupportedModalityError):
        await enrich_content_items([_img_item()])


async def test_enrich_transient_llm_error_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aparse(raw_file: Any) -> ParsedContent:
        raise LLMError("provider down")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    items = [_img_item()]
    await enrich_content_items(items)

    assert items[0]["parse_status"] == "failed"
    assert "parsed_content" not in items[0]


async def test_enrich_html_base64_routes_as_html_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_aparse(raw_file: Any) -> ParsedContent:
        seen["extension"] = raw_file.extension
        seen["content"] = raw_file.content
        return ParsedContent(text="HTML PARSED")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    items = [_html_b64_item()]
    await enrich_content_items(items)

    assert items[0]["parsed_content"] == "HTML PARSED"
    assert items[0]["parse_status"] == "success"
    assert seen["extension"] == "html"
    assert b"v9.9.9" in seen["content"]


async def test_enrich_http_uri_routes_as_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_aparse(raw_file: Any) -> ParsedContent:
        seen["uri"] = raw_file.uri
        seen["content"] = raw_file.content
        return ParsedContent(text="URL PARSED")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    items = [_html_uri_item()]
    await enrich_content_items(items)

    assert items[0]["parsed_content"] == "URL PARSED"
    assert items[0]["parse_status"] == "success"
    assert seen["uri"] == "https://example.com/page.html"
    assert seen["content"] == b""


async def test_enrich_html_text_only_raises_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aparse(raw_file: Any) -> ParsedContent:
        return ParsedContent(text="should-not-be-reached")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    with pytest.raises(UnsupportedModalityError):
        await enrich_content_items([{"type": "html", "text": "<p>hi</p>"}])


async def test_enrich_file_uri_hydrates_and_parses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_aparse(raw_file: Any) -> ParsedContent:
        seen["content"] = raw_file.content
        seen["uri"] = raw_file.uri
        return ParsedContent(text="FILE PARSED")

    monkeypatch.setattr(_APARSE_FILE_TARGET, fake_aparse)
    f = tmp_path / "doc.html"
    f.write_bytes(b"<html>hello</html>")
    items = [{"type": "html", "uri": f"file://{f}"}]
    await enrich_content_items(items)

    assert items[0]["parsed_content"] == "FILE PARSED"
    assert items[0]["parse_status"] == "success"
    assert seen["content"] == b"<html>hello</html>"
    assert seen["uri"] == ""
