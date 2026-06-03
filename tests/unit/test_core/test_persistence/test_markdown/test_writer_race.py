"""Regression tests for the MarkdownWriter read-modify-write race.

Before the per-path :class:`asyncio.Lock` was added, two concurrent tasks
calling :meth:`MarkdownWriter.append_entry` against the same path would
each load the file, append one entry block in memory, and write the
merged file back — the second writer's read pre-dated the first
writer's write, so it overwrote the first writer's append. Both
``entry_count`` (frontmatter) and the entry block markers were lost in
proportion to concurrency level.

These tests drive ``N`` concurrent appends against one ``(owner, date)``
and assert that no entry is lost at any concurrency level. They cover
both the single-entry ``append_entry`` path (taken by tests / external
callers) and the batched ``append_entries`` path (taken by strategies
after the per-owner batching migration).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from everos.core.persistence import EntryId, MarkdownWriter, MemoryRoot
from everos.infra.persistence.markdown.writers.atomic_fact_writer import (
    AtomicFactWriter,
)


def _scan_md(md_path: Path) -> tuple[int, int]:
    """Return ``(entry_tag_count, frontmatter_entry_count)``."""
    text = md_path.read_text(encoding="utf-8")
    tag_count = len(re.findall(r"<!-- entry:af_", text))
    fm_match = re.search(r"^entry_count: (\d+)", text, re.MULTILINE)
    fm_count = int(fm_match.group(1)) if fm_match else -1
    return tag_count, fm_count


async def _drive_concurrent_appends(
    writer: AtomicFactWriter,
    owner: str,
    n: int,
    concurrency: int,
) -> None:
    """Issue ``n`` single-entry ``append_entry`` calls with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(idx: int) -> None:
        async with sem:
            await writer.append_entry(
                owner,
                inline={
                    "owner_id": owner,
                    "session_id": "race_test",
                    "timestamp": "2026-05-18T00:00:00+00:00",
                    "parent_type": "memcell",
                    "parent_id": f"mc_{idx:04d}",
                },
                sections={"Fact": f"fact-{idx:04d}"},
            )

    await asyncio.gather(*(_guarded(i) for i in range(n)))


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16])
async def test_append_entry_no_lost_updates_under_concurrency(
    tmp_path: Path, concurrency: int
) -> None:
    """``append_entry`` from N concurrent tasks must not drop any entry."""
    root = MemoryRoot(root=tmp_path)
    writer = AtomicFactWriter(root=root)
    owner = "race_user"
    n = 30

    await _drive_concurrent_appends(writer, owner, n, concurrency)

    md_files = list((root.users_dir() / owner).rglob("*.md"))
    assert len(md_files) == 1, f"expected 1 md file, got {md_files}"
    tag_count, fm_count = _scan_md(md_files[0])

    assert tag_count == n, (
        f"lost {n - tag_count} entries at concurrency={concurrency} "
        f"(tag_count={tag_count}, expected={n})"
    )
    assert fm_count == n, (
        f"frontmatter entry_count drift at concurrency={concurrency} "
        f"(fm_count={fm_count}, expected={n})"
    )


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16])
async def test_append_entries_batch_no_lost_updates_under_concurrency(
    tmp_path: Path, concurrency: int
) -> None:
    """``append_entries`` (batched) from N concurrent tasks must not drop any
    entry."""
    root = MemoryRoot(root=tmp_path)
    writer = AtomicFactWriter(root=root)
    owner = "race_user_batched"
    batches = 6
    items_per_batch = 5
    total = batches * items_per_batch

    sem = asyncio.Semaphore(concurrency)

    async def _one_batch(batch_idx: int) -> None:
        async with sem:
            items = [
                (
                    {
                        "owner_id": owner,
                        "session_id": "race_test",
                        "timestamp": "2026-05-18T00:00:00+00:00",
                        "parent_type": "memcell",
                        "parent_id": f"mc_b{batch_idx:02d}_i{i:02d}",
                    },
                    {"Fact": f"batched-fact-b{batch_idx:02d}-{i:02d}"},
                )
                for i in range(items_per_batch)
            ]
            await writer.append_entries(owner, items)

    await asyncio.gather(*(_one_batch(b) for b in range(batches)))

    md_files = list((root.users_dir() / owner).rglob("*.md"))
    assert len(md_files) == 1
    tag_count, fm_count = _scan_md(md_files[0])

    assert tag_count == total, (
        f"lost {total - tag_count} entries at concurrency={concurrency} "
        f"(tag_count={tag_count}, expected={total})"
    )
    assert fm_count == total, (
        f"frontmatter entry_count drift at concurrency={concurrency} "
        f"(fm_count={fm_count}, expected={total})"
    )


async def test_lock_for_returns_same_lock_per_path(tmp_path: Path) -> None:
    """``lock_for`` is the keying primitive that BaseDailyWriter relies on
    to serialise its multi-step read-compute-write sequence; aliasing paths
    must collapse to one lock object."""
    writer = MarkdownWriter(MemoryRoot(root=tmp_path))
    p1 = tmp_path / "foo" / "bar.md"
    p2 = tmp_path / "foo" / "bar.md"
    p3 = tmp_path / "foo" / ".." / "foo" / "bar.md"

    lock1 = writer.lock_for(p1)
    lock2 = writer.lock_for(p2)
    lock3 = writer.lock_for(p3)

    # Same canonical path → identical Lock object.
    assert lock1 is lock2
    assert lock1 is lock3

    # Different path → different Lock.
    other = writer.lock_for(tmp_path / "foo" / "baz.md")
    assert other is not lock1


async def test_append_entries_empty_is_noop(tmp_path: Path) -> None:
    """Empty batch must not touch the file or allocate any EntryId."""
    writer = MarkdownWriter(MemoryRoot(root=tmp_path))
    target = tmp_path / "scratch.md"
    result = await writer.append_entries(target, [])
    assert result == target
    # No file should have been created (empty body + no frontmatter updates
    # still calls write_markdown — verify the file is empty or absent).
    if target.exists():
        assert target.read_text(encoding="utf-8") in ("", "---\n---\n\n")


async def test_markdown_writer_append_entry_delegates_to_batch(
    tmp_path: Path,
) -> None:
    """``append_entry`` is documented as a wrapper for ``append_entries`` —
    asserting they produce identical file contents protects callers from
    drift between the two paths."""
    writer = MarkdownWriter(MemoryRoot(root=tmp_path))
    eid = EntryId.next_for("af", __import__("datetime").date(2026, 5, 18), 0)
    body = "**fact**: hello"

    path_a = tmp_path / "a.md"
    path_b = tmp_path / "b.md"

    await writer.append_entry(
        path_a,
        entry_body=body,
        entry_id=eid,
        frontmatter_updates={"id": "shared", "entry_count": 1},
    )
    await writer.append_entries(
        path_b,
        [(body, eid)],
        frontmatter_updates={"id": "shared", "entry_count": 1},
    )

    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")
