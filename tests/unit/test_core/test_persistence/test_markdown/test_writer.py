"""Unit tests for MarkdownWriter (atomic write)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import patch

import pytest

from everos.core.persistence import (
    EntryId,
    MarkdownReader,
    MarkdownWriter,
    MemoryRoot,
)


def _make_writer(tmp_path: Path) -> MarkdownWriter:
    return MarkdownWriter(MemoryRoot(tmp_path))


async def test_write_creates_file_with_content(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "users" / "u1" / "out.md"
    result = await writer.write(target, "hello\n")
    assert result == target
    assert target.read_text(encoding="utf-8") == "hello\n"


async def test_write_creates_parent_directories(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "a" / "b" / "c" / "f.md"
    await writer.write(target, "x")
    assert target.is_file()


async def test_write_overwrites_existing(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "f.md"
    target.write_text("old", encoding="utf-8")
    await writer.write(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


async def test_write_no_temp_file_left_after_success(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "f.md"
    await writer.write(target, "ok")
    leftovers = [
        p.name
        for p in tmp_path.iterdir()  # noqa: ASYNC240 — sync iterdir over a pytest tmp_path is fine in tests
        if p.name.startswith(".f.md.tmp.")
    ]
    assert leftovers == []


async def test_write_cleans_up_temp_on_failure(tmp_path: Path) -> None:
    """If os.replace fails, the temp file should be cleaned up."""
    writer = _make_writer(tmp_path)
    target = tmp_path / "f.md"

    boom = OSError("simulated rename failure")
    with (
        patch("everos.core.persistence.markdown.writer.os.replace", side_effect=boom),
        pytest.raises(OSError, match="simulated"),
    ):
        await writer.write(target, "hello")

    # No tmp file leftover, and the target was not created.
    leftovers = [
        p.name
        for p in tmp_path.iterdir()  # noqa: ASYNC240 — sync iterdir over a pytest tmp_path is fine in tests
        if p.name.startswith(".f.md.tmp.")
    ]
    assert leftovers == []
    assert not target.exists()


async def test_write_markdown_assembles_frontmatter_and_body(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "doc.md"
    await writer.write_markdown(
        target,
        frontmatter={"title": "Hello"},
        body="# Body\n",
    )
    text = target.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "title: Hello" in text
    assert text.rstrip("\n").endswith("# Body")


async def test_write_markdown_round_trip(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "rt.md"
    await writer.write_markdown(
        target,
        frontmatter={"k": "v", "n": 1},
        body="<!-- entry:x -->\ncontent\n<!-- /entry:x -->\n",
    )
    parsed = await MarkdownReader.read(target)
    assert parsed.frontmatter == {"k": "v", "n": 1}
    assert len(parsed.entries) == 1
    assert parsed.entries[0].body == "content"


async def test_write_markdown_no_frontmatter(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "plain.md"
    await writer.write_markdown(target, body="just body\n")
    assert target.read_text(encoding="utf-8") == "just body\n"


def test_memory_root_property_accessible(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    assert writer.memory_root.root == tmp_path.resolve()


# ── append_entry ─────────────────────────────────────────────────────────


async def test_append_entry_creates_file_when_missing(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "memcells" / "memcell-2026-04-22.md"
    eid = EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=1)
    written = await writer.append_entry(
        target,
        entry_body="hello world",
        entry_id=eid,
        frontmatter_updates={
            "file_type": "memcell_daily",
            "entry_count": 1,
        },
    )
    assert written == target
    parsed = await MarkdownReader.read(target)
    assert parsed.frontmatter == {"file_type": "memcell_daily", "entry_count": 1}
    assert len(parsed.entries) == 1
    assert parsed.entries[0].id == "umc_20260422_00000001"
    assert parsed.entries[0].body == "hello world"


async def test_append_entry_appends_to_existing(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "log.md"
    await writer.append_entry(
        target,
        entry_body="first",
        entry_id=EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=1),
        frontmatter_updates={"entry_count": 1},
    )
    await writer.append_entry(
        target,
        entry_body="second",
        entry_id=EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=2),
        frontmatter_updates={"entry_count": 2},
    )
    parsed = await MarkdownReader.read(target)
    assert [e.id for e in parsed.entries] == [
        "umc_20260422_00000001",
        "umc_20260422_00000002",
    ]
    assert [e.body for e in parsed.entries] == ["first", "second"]


async def test_append_entry_merges_frontmatter_shallow(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "log.md"
    await writer.append_entry(
        target,
        entry_body="b",
        entry_id=EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=1),
        frontmatter_updates={
            "file_type": "memcell_daily",
            "entry_count": 1,
            "last_appended_at": "2026-04-22T10:00:00Z",
        },
    )
    # Second append — overwrite entry_count + last_appended_at, keep file_type.
    await writer.append_entry(
        target,
        entry_body="b",
        entry_id=EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=2),
        frontmatter_updates={
            "entry_count": 2,
            "last_appended_at": "2026-04-22T10:05:00Z",
        },
    )
    parsed = await MarkdownReader.read(target)
    assert parsed.frontmatter == {
        "file_type": "memcell_daily",
        "entry_count": 2,
        "last_appended_at": "2026-04-22T10:05:00Z",
    }


async def test_append_entry_without_frontmatter_updates_keeps_existing(
    tmp_path: Path,
) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "log.md"
    await writer.write_markdown(target, frontmatter={"file_type": "x", "n": 1}, body="")
    await writer.append_entry(
        target,
        entry_body="body",
        entry_id=EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=1),
    )
    parsed = await MarkdownReader.read(target)
    assert parsed.frontmatter == {"file_type": "x", "n": 1}
    assert len(parsed.entries) == 1


async def test_append_entry_round_trip_with_reader(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    target = tmp_path / "log.md"
    for i in range(5):
        await writer.append_entry(
            target,
            entry_body=f"content {i}",
            entry_id=EntryId(prefix="umc", date=dt.date(2026, 4, 22), seq=i + 1),
            frontmatter_updates={"entry_count": i + 1},
        )
    parsed = await MarkdownReader.read(target)
    assert len(parsed.entries) == 5
    assert parsed.frontmatter["entry_count"] == 5
    for i, e in enumerate(parsed.entries):
        assert e.id == f"umc_20260422_{i + 1:08d}"
        assert e.body == f"content {i}"
