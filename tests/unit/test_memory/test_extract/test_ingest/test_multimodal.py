"""Tests for ingest content coercion + text derivation (tagged rendering)."""

from __future__ import annotations

from everos.memory.extract.ingest.multimodal import (
    coerce_items,
    derive_text,
    normalise_content,
)


def test_coerce_str_to_text_item() -> None:
    assert coerce_items("hi") == [{"type": "text", "text": "hi"}]


def test_derive_text_renders_parsed_nontext_as_tag() -> None:
    items = [
        {"type": "text", "text": "before"},
        {"type": "image", "name": "p.png", "parsed_content": "OCR TEXT"},
        {"type": "text", "text": "after"},
    ]
    text, non_text = derive_text(items)

    assert "[IMAGE: p.png]\nOCR TEXT" in text
    assert text.startswith("before")
    assert text.endswith("after")
    assert non_text == 0


def test_derive_text_counts_unparsed_nontext() -> None:
    text, non_text = derive_text([{"type": "image", "uri": "x"}])
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
