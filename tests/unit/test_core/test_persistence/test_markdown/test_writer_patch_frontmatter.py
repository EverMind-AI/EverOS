"""Unit tests for MarkdownWriter.patch_frontmatter."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from everos.core.persistence import (
    EntryId,
    MarkdownReader,
    MarkdownWriter,
    MemoryRoot,
)


def _make_writer(tmp_path: Path) -> MarkdownWriter:
    return MarkdownWriter(MemoryRoot(tmp_path))


async def test_patch_frontmatter_adds_new_field(tmp_path: Path) -> None:
    """Patching a field that does not exist in frontmatter adds it."""
    writer = _make_writer(tmp_path)
    target = tmp_path / "doc.md"

    await writer.write_markdown(
        target,
        frontmatter={"type": "episode_daily", "entry_count": 0},
        body="",
    )
    await writer.patch_frontmatter(target, {"new_key": "new_val"})

    parsed = await MarkdownReader.read(target)
    assert parsed.frontmatter["new_key"] == "new_val"
    assert parsed.frontmatter["type"] == "episode_daily"
    assert parsed.frontmatter["entry_count"] == 0


async def test_patch_frontmatter_updates_existing_field(tmp_path: Path) -> None:
    """Patching an existing scalar field overwrites it."""
    writer = _make_writer(tmp_path)
    target = tmp_path / "doc.md"

    await writer.write_markdown(
        target,
        frontmatter={"type": "episode_daily", "entry_count": 3},
        body="",
    )
    await writer.patch_frontmatter(target, {"entry_count": 5})

    parsed = await MarkdownReader.read(target)
    assert parsed.frontmatter["entry_count"] == 5
    assert parsed.frontmatter["type"] == "episode_daily"


async def test_patch_frontmatter_merges_dict_field(tmp_path: Path) -> None:
    """Dict fields are merged additively, not replaced wholesale."""
    writer = _make_writer(tmp_path)
    target = tmp_path / "doc.md"

    await writer.write_markdown(
        target,
        frontmatter={
            "type": "episode_daily",
            "deprecated_entries": {"ep_001": "ep_100"},
        },
        body="",
    )
    # Merge a second entry — the first must survive.
    await writer.patch_frontmatter(target, {"deprecated_entries": {"ep_002": "ep_101"}})

    parsed = await MarkdownReader.read(target)
    dep = parsed.frontmatter["deprecated_entries"]
    assert dep == {"ep_001": "ep_100", "ep_002": "ep_101"}


async def test_patch_frontmatter_preserves_entries(tmp_path: Path) -> None:
    """Entry blocks in the body must be byte-identical after a patch."""
    writer = _make_writer(tmp_path)
    target = tmp_path / "doc.md"

    eid1 = EntryId(prefix="ep", date=dt.date(2026, 5, 1), seq=1)
    eid2 = EntryId(prefix="ep", date=dt.date(2026, 5, 1), seq=2)
    await writer.append_entries(
        target,
        [("first entry body", eid1), ("second entry body", eid2)],
        frontmatter_updates={"type": "episode_daily", "entry_count": 2},
    )

    # Snapshot entries before patch.
    pre = await MarkdownReader.read(target)
    pre_ids = [e.id for e in pre.entries]
    pre_bodies = [e.body for e in pre.entries]

    # Patch frontmatter only.
    await writer.patch_frontmatter(target, {"deprecated_entries": {"ep_001": "ep_999"}})

    # Entries must survive unchanged.
    post = await MarkdownReader.read(target)
    assert [e.id for e in post.entries] == pre_ids
    assert [e.body for e in post.entries] == pre_bodies
    assert post.frontmatter["deprecated_entries"] == {"ep_001": "ep_999"}
    assert post.frontmatter["entry_count"] == 2
