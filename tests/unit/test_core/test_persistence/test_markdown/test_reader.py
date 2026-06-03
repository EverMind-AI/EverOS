"""Unit tests for MarkdownReader."""

from __future__ import annotations

import datetime
from pathlib import Path

from everos.core.persistence import MarkdownReader


def test_parse_text_with_frontmatter_and_entries() -> None:
    text = (
        "---\n"
        "title: Day Log\n"
        "date: 2026-04-22\n"
        "---\n"
        "# Header\n"
        "<!-- entry:e1 -->\n"
        "first entry\n"
        "<!-- /entry:e1 -->\n"
    )
    parsed = MarkdownReader.parse(text)
    # PyYAML auto-converts unquoted ISO dates to datetime.date.
    assert parsed.frontmatter == {
        "title": "Day Log",
        "date": datetime.date(2026, 4, 22),
    }
    assert "# Header" in parsed.body
    assert len(parsed.entries) == 1
    assert parsed.entries[0].id == "e1"
    assert parsed.entries[0].body == "first entry"


def test_parse_no_frontmatter_no_entries() -> None:
    text = "# Just a header\n\nbody.\n"
    parsed = MarkdownReader.parse(text)
    assert parsed.frontmatter == {}
    assert parsed.body == text
    assert parsed.entries == []


def test_parse_only_frontmatter() -> None:
    text = "---\nkey: value\n---\n"
    parsed = MarkdownReader.parse(text)
    assert parsed.frontmatter == {"key": "value"}
    assert parsed.body == ""
    assert parsed.entries == []


async def test_read_file(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(
        "---\nk: v\n---\n<!-- entry:x -->\nbody\n<!-- /entry:x -->\n",
        encoding="utf-8",
    )
    parsed = await MarkdownReader.read(f)
    assert parsed.frontmatter == {"k": "v"}
    assert parsed.entries[0].id == "x"


async def test_read_unicode_file(tmp_path: Path) -> None:
    f = tmp_path / "zh.md"
    f.write_text("---\ntitle: 你好\n---\n世界\n", encoding="utf-8")
    parsed = await MarkdownReader.read(f)
    assert parsed.frontmatter == {"title": "你好"}
    assert parsed.body == "世界\n"
