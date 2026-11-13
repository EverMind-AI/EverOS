"""Unit tests for MarkdownReader."""

from __future__ import annotations

import datetime
from pathlib import Path

import structlog.testing

from everos.infra.persistence.markdown import MarkdownReader


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


_BROKEN = (
    "---\n"
    "entry_count: 3\n"
    "---\n"
    "<!-- entry:e1 -->\n"
    "first\n"
    "<!-- /entry:e1 -->\n"
    "<!-- entry:e2 -->\n"
    "second, its close marker deleted\n"
    "<!-- entry:e3 -->\n"
    "third\n"
    "<!-- /entry:e3 -->\n"
)


def test_parse_names_an_unclosed_entry_and_reads_the_rest() -> None:
    parsed = MarkdownReader.parse(_BROKEN)

    assert [e.id for e in parsed.entries] == ["e1", "e3"]
    assert [(u.id, u.reason) for u in parsed.unreadable] == [
        ("e2", "no closing marker")
    ]
    # Lines are the file's, frontmatter included.
    assert parsed.body_line == 4
    assert parsed.line_of(parsed.unreadable[0].start) == 7
    assert parsed.line_of(parsed.entries[1].start) == 9


async def test_read_logs_each_unclosed_entry(tmp_path: Path) -> None:
    path = tmp_path / "log.md"
    path.write_text(_BROKEN, encoding="utf-8")

    with structlog.testing.capture_logs() as captured:
        parsed = await MarkdownReader.read(path)

    assert [e.id for e in parsed.entries] == ["e1", "e3"]
    warned = [e for e in captured if e["event"] == "markdown_entry_unreadable"]
    assert [(e["path"], e["entry_id"], e["line"], e["reason"]) for e in warned] == [
        (str(path), "e2", 7, "no closing marker")
    ]
    assert warned[0]["log_level"] == "warning"


async def test_read_of_a_well_formed_file_logs_nothing(tmp_path: Path) -> None:
    path = tmp_path / "log.md"
    path.write_text(_BROKEN.replace("deleted\n", "deleted\n<!-- /entry:e2 -->\n"))

    with structlog.testing.capture_logs() as captured:
        parsed = await MarkdownReader.read(path)

    assert [e.id for e in parsed.entries] == ["e1", "e2", "e3"]
    assert parsed.unreadable == []
    assert captured == []
