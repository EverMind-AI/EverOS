"""Tests for enrich_content_items (everalgo.parser.aparse is monkeypatched)."""

from __future__ import annotations

import base64
from typing import Any

import pytest

# ``everalgo.parser`` ships under the ``[multimodal]`` extra (see
# pyproject.toml). CI doesn't install that extra by default, and these
# tests monkeypatch ``everalgo.parser.aparse`` — which requires the
# module to actually be importable, otherwise ``monkeypatch.setattr``
# fails at resolve-time. Skip the whole module when the optional
# dependency isn't present; we still run when ``multimodal`` is installed.
pytest.importorskip("everalgo.parser")

from everalgo.llm import LLMError  # noqa: E402
from everalgo.types import ParsedContent  # noqa: E402

from everos.core.errors import UnsupportedModalityError  # noqa: E402
from everos.memory.extract.parser import enrich_content_items  # noqa: E402


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
    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        return ParsedContent(text="OCR RESULT")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    items: list[dict[str, Any]] = [{"type": "text", "text": "hi"}, _img_item()]
    await enrich_content_items(items, llm=object(), max_concurrency=2)

    assert items[1]["parsed_content"] == "OCR RESULT"
    assert items[1]["parse_status"] == "success"
    assert "parsed_content" not in items[0]  # text item untouched


async def test_enrich_unsupported_modality_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        raise NotImplementedError("video deferred")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    with pytest.raises(UnsupportedModalityError):
        await enrich_content_items([_img_item()], llm=object())


async def test_enrich_transient_llm_error_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        raise LLMError("provider down")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    items = [_img_item()]
    await enrich_content_items(items, llm=object())  # must not raise

    assert items[0]["parse_status"] == "failed"
    assert "parsed_content" not in items[0]


async def test_enrich_html_base64_routes_as_html_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A type=html base64 item reaches the parser as html-extension bytes.

    Locks the "normal HTML file call" contract: base64 + ext=html maps to
    a RawFile the parser dispatches as HTML (vs the 415 that a text-only
    html item produces — see test_ingest for that negative path).
    """
    seen: dict[str, Any] = {}

    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        seen["extension"] = raw_file.extension
        seen["content"] = raw_file.content
        return ParsedContent(text="HTML PARSED")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    items = [_html_b64_item()]
    await enrich_content_items(items, llm=object())

    assert items[0]["parsed_content"] == "HTML PARSED"
    assert items[0]["parse_status"] == "success"
    assert seen["extension"] == "html"
    assert b"v9.9.9" in seen["content"]


async def test_enrich_http_uri_routes_as_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An http(s) uri item reaches the parser as a uri RawFile (no bytes).

    Proves everos forwards uri-backed items to the parser, which is what
    drives everalgo's URL-fetch dispatch path (http/https only; file:// is
    rejected downstream).
    """
    seen: dict[str, Any] = {}

    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        seen["uri"] = raw_file.uri
        seen["content"] = raw_file.content
        return ParsedContent(text="URL PARSED")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    items = [_html_uri_item()]
    await enrich_content_items(items, llm=object())

    assert items[0]["parsed_content"] == "URL PARSED"
    assert items[0]["parse_status"] == "success"
    assert seen["uri"] == "https://example.com/page.html"
    assert seen["content"] == b""


async def test_enrich_html_text_only_raises_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """type=html carrying only ``text`` (no uri/base64) is undispatchable.

    Any non-text item is routed to the parser, which needs a fetchable or
    decodable payload; a bare ``text`` has neither, so it surfaces as a
    MultimodalError (the route maps it to HTTP 415). To inline HTML *as
    text*, callers must use ``type="text"`` instead.
    """

    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        return ParsedContent(text="should-not-be-reached")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    with pytest.raises(UnsupportedModalityError):
        await enrich_content_items(
            [{"type": "html", "text": "<p>hi</p>"}], llm=object()
        )


async def test_enrich_file_uri_hydrates_and_parses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """A ``file://`` item is read locally and handed to the parser as bytes.

    Proves EverOS hydrates the file (everalgo never sees the path / fs) — the
    parser receives ``content`` bytes, not a uri.
    """
    seen: dict[str, Any] = {}

    async def fake_aparse(raw_file: Any, *, llm: Any) -> ParsedContent:
        seen["content"] = raw_file.content
        seen["uri"] = raw_file.uri
        return ParsedContent(text="FILE PARSED")

    monkeypatch.setattr("everalgo.parser.aparse", fake_aparse)
    f = tmp_path / "doc.html"
    f.write_bytes(b"<html>hello</html>")
    items = [{"type": "html", "uri": f"file://{f}"}]
    await enrich_content_items(items, llm=object())

    assert items[0]["parsed_content"] == "FILE PARSED"
    assert items[0]["parse_status"] == "success"
    assert seen["content"] == b"<html>hello</html>"  # hydrated, not a pointer
    assert seen["uri"] == ""
