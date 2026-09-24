"""Watcher behaviour on a real filesystem, asserted in the state table.

The pure helpers are covered in ``test_watcher_helpers.py``. This module
covers everything else: the four ``_Handler`` callbacks, the two drop paths
in ``_enqueue``, ``start()`` creating the root, and -- the part only a real
observer can prove -- that this OS's filesystem events reach
``md_change_state`` at all.

That last part is what makes the Windows CI job an oracle. inotify, FSEvents
and ReadDirectoryChangesW disagree about what an editor's save looks like.
Windows reports ``os.replace`` over an existing file as a REMOVED of the
target followed by a RENAMED pair, so ``on_deleted`` fires for a path that
still exists -- the shape that, unguarded, drives the worker to
``delete_by_md_path`` and wipes LanceDB while the md is fine. Every assertion
here is on the final row, never on event order, so one test holds on all
three backends.

Assertions read ``md_change_state`` directly; that table owns the fact.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from sqlmodel import SQLModel, select
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from everos.core.persistence import MemoryRoot
from everos.core.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite import (
    MdChangeState,
    dispose_engine,
    get_engine,
    get_session_factory,
    md_change_state_repo,
)
from everos.memory.cascade import watcher as watcher_mod
from everos.memory.cascade.registry import match_kind
from everos.memory.cascade.scanner import CascadeScanner
from everos.memory.cascade.watcher import CascadeWatcher, _enqueue_async, _Handler

_EPISODE_DIR = ("default_app", "default_project", "users", "u1", "episodes")
# windows-latest delivers in about a second; the rest is margin for a loaded
# runner. A passing test never waits this long.
_EVENT_DEADLINE_S = 15.0
# Long enough for a trailing REMOVED / RENAMED leg to land after the row first
# appeared, so a "never deleted" assertion is not just "not deleted yet".
_TRAILING_EVENT_GRACE_S = 1.5


@pytest.fixture
async def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    """Boot the system db against a tmp memory_root; no LanceDB needed here."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    mr = MemoryRoot.resolve()
    mr.ensure()
    yield mr
    await dispose_engine()


@pytest.fixture
async def watcher(runtime: MemoryRoot) -> AsyncIterator[CascadeWatcher]:
    """A real observer on the memory root -- inotify / FSEvents / Win32."""
    w = CascadeWatcher(runtime, asyncio.get_running_loop())
    w.start()
    # Let the native observer thread arm before the test writes, so the very
    # first event is not lost to a race with startup.
    await asyncio.sleep(0.5)
    yield w
    w.stop()


def _episode(root: Path, name: str = "episode-2026-01-01.md") -> Path:
    d = root.joinpath(*_EPISODE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _rel(root: Path, p: Path) -> str:
    return p.resolve().relative_to(root).as_posix()


async def _row(md_path: str) -> MdChangeState | None:
    async with session_scope(get_session_factory()) as s:
        stmt = select(MdChangeState).where(MdChangeState.md_path == md_path)
        return (await s.execute(stmt)).scalars().first()


async def _all_paths() -> list[str]:
    async with session_scope(get_session_factory()) as s:
        rows = (await s.execute(select(MdChangeState))).scalars().all()
        return sorted(r.md_path for r in rows)


async def _wait_row(
    md_path: str,
    where: Callable[[MdChangeState], bool] = lambda _r: True,
    *,
    deadline: float = _EVENT_DEADLINE_S,
) -> MdChangeState:
    """Poll for a row matching ``where``; on timeout say what IS there."""
    end = time.monotonic() + deadline
    last: MdChangeState | None = None
    while time.monotonic() < end:
        last = await _row(md_path)
        if last is not None and where(last):
            return last
        await asyncio.sleep(0.1)
    seen = f"row is {last.change_type!r}/{last.status!r}" if last else "no row"
    pytest.fail(
        f"{md_path!r}: {seen} after {deadline}s; rows present: {await _all_paths()}"
    )


async def _settle() -> None:
    """Let a ``run_coroutine_threadsafe`` hop land on this loop."""
    for _ in range(5):
        await asyncio.sleep(0.02)


# ── _Handler callbacks, called directly ─────────────────────────────────────


async def test_created_enqueues_added(runtime: MemoryRoot) -> None:
    p = _episode(runtime.root)
    p.write_text("x", encoding="utf-8")
    _Handler(runtime, asyncio.get_running_loop()).on_created(FileCreatedEvent(str(p)))
    await _settle()
    row = await _wait_row(_rel(runtime.root, p), deadline=2)
    assert (row.kind, row.change_type, row.status) == ("episode", "added", "pending")
    assert row.mtime > 0


async def test_modified_enqueues_modified(runtime: MemoryRoot) -> None:
    p = _episode(runtime.root)
    p.write_text("x", encoding="utf-8")
    _Handler(runtime, asyncio.get_running_loop()).on_modified(FileModifiedEvent(str(p)))
    await _settle()
    row = await _wait_row(_rel(runtime.root, p), deadline=2)
    assert row.change_type == "modified"


async def test_deleted_for_a_path_that_still_exists_is_ignored(
    runtime: MemoryRoot,
) -> None:
    """The LanceDB-wipe guard.

    FSEvents (``os.replace``) and ReadDirectoryChangesW (REMOVED of an
    overwritten target) both hand the handler a deletion for a path that is
    still there. Propagating it would enqueue ``deleted`` and the worker would
    drop the rows for a file that is intact.
    """
    p = _episode(runtime.root)
    p.write_text("still here", encoding="utf-8")
    _Handler(runtime, asyncio.get_running_loop()).on_deleted(FileDeletedEvent(str(p)))
    await _settle()
    assert await _row(_rel(runtime.root, p)) is None


async def test_deleted_for_a_gone_path_enqueues_deleted_with_zero_mtime(
    runtime: MemoryRoot,
) -> None:
    p = _episode(runtime.root)  # directory exists, file never written
    _Handler(runtime, asyncio.get_running_loop()).on_deleted(FileDeletedEvent(str(p)))
    await _settle()
    row = await _wait_row(_rel(runtime.root, p), deadline=2)
    assert (row.change_type, row.mtime) == ("deleted", 0.0)


async def test_modified_for_a_gone_path_is_recorded_as_deleted(
    runtime: MemoryRoot,
) -> None:
    """A stale ``modified`` must not resurrect a deleted file.

    FSEvents can deliver the modified leg of a create after the unlink that
    followed it (see ``test_unlink_enqueues_deleted``, which flaked 2 in 6 on
    macOS before this guard). Disk is the truth: a modification reported for
    a path that is no longer there is a deletion.
    """
    p = _episode(runtime.root)  # directory exists, file never written
    _Handler(runtime, asyncio.get_running_loop()).on_modified(FileModifiedEvent(str(p)))
    await _settle()
    row = await _wait_row(_rel(runtime.root, p), deadline=2)
    assert (row.change_type, row.mtime) == ("deleted", 0.0)


async def test_moved_enqueues_source_deleted_and_dest_added(
    runtime: MemoryRoot,
) -> None:
    src = _episode(runtime.root, "episode-2026-01-01.md")  # not on disk
    dest = _episode(runtime.root, "episode-2026-01-02.md")
    dest.write_text("moved", encoding="utf-8")
    _Handler(runtime, asyncio.get_running_loop()).on_moved(
        FileMovedEvent(str(src), str(dest))
    )
    await _settle()
    assert (await _wait_row(_rel(runtime.root, src), deadline=2)).change_type == (
        "deleted"
    )
    assert (await _wait_row(_rel(runtime.root, dest), deadline=2)).change_type == (
        "added"
    )


async def test_moved_keeps_source_when_it_still_exists(runtime: MemoryRoot) -> None:
    """A hardlink survives the rename, so the named path is still bound."""
    src = _episode(runtime.root, "episode-2026-01-01.md")
    dest = _episode(runtime.root, "episode-2026-01-02.md")
    src.write_text("linked", encoding="utf-8")
    try:
        os.link(src, dest)
    except OSError as exc:  # filesystem without hardlinks
        pytest.skip(f"hardlinks unavailable here: {exc}")
    _Handler(runtime, asyncio.get_running_loop()).on_moved(
        FileMovedEvent(str(src), str(dest))
    )
    await _settle()
    assert (await _wait_row(_rel(runtime.root, dest), deadline=2)).change_type == (
        "added"
    )
    assert await _row(_rel(runtime.root, src)) is None


async def test_path_outside_root_is_dropped(runtime: MemoryRoot) -> None:
    outside = runtime.root.parent.joinpath("elsewhere", *_EPISODE_DIR, "episode-x.md")
    _Handler(runtime, asyncio.get_running_loop()).on_created(
        FileCreatedEvent(str(outside))
    )
    await _settle()
    assert await _all_paths() == []


async def test_path_not_matching_a_kind_is_dropped(
    runtime: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropped *before* the async hop, not lost inside it.

    "No row" alone cannot tell a clean drop from a failure downstream: an
    exception inside ``_enqueue_async`` -- logged, or left unretrieved in the
    ``run_coroutine_threadsafe`` future -- also leaves no row. Recording what
    the guard hands downstream is what separates the two.
    """
    handed_down: list[tuple[object, ...]] = []

    async def record(*a: object, **k: object) -> None:
        handed_down.append(a)

    monkeypatch.setattr(watcher_mod, "_enqueue_async", record)
    p = runtime.root / "notes" / "random.md"
    p.parent.mkdir(parents=True)
    p.write_text("x", encoding="utf-8")
    _Handler(runtime, asyncio.get_running_loop()).on_created(FileCreatedEvent(str(p)))
    await _settle()
    assert handed_down == [], "a path with no kind reached the enqueue coroutine"
    assert await _all_paths() == []


async def test_upsert_failure_is_logged_not_raised(
    runtime: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The callback runs on the watchdog thread; an escape would kill it."""

    async def boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("sqlite is having a day")

    monkeypatch.setattr(md_change_state_repo, "upsert", boom)
    spec = match_kind("/".join((*_EPISODE_DIR, "episode-x.md")))
    assert spec is not None
    await _enqueue_async(spec, "whatever.md", "added", 1.0)  # must not raise


async def test_start_creates_a_missing_root_and_stop_is_clean(tmp_path: Path) -> None:
    """watchdog refuses a non-existent path; ``start()`` has to make it first."""
    mr = MemoryRoot(tmp_path / "not-yet")
    assert not mr.root.exists()
    w = CascadeWatcher(mr, asyncio.get_running_loop())
    w.start()
    try:
        assert mr.root.is_dir()
    finally:
        w.stop()


# ── a real observer: does this OS deliver at all? ───────────────────────────


async def test_observer_delivers_a_new_file(
    runtime: MemoryRoot, watcher: CascadeWatcher
) -> None:
    p = _episode(runtime.root)
    p.write_text("hello", encoding="utf-8")
    row = await _wait_row(_rel(runtime.root, p))
    assert row.kind == "episode"
    assert row.change_type in {"added", "modified"}
    assert row.status == "pending"
    assert row.mtime > 0


async def test_in_place_save_never_registers_as_deleted(
    runtime: MemoryRoot, watcher: CascadeWatcher
) -> None:
    """Notepad / VS Code / Obsidian truncate and rewrite the same inode."""
    p = _episode(runtime.root)
    p.write_text("v1", encoding="utf-8")
    rel = _rel(runtime.root, p)
    await _wait_row(rel)
    first = await _row(rel)
    assert first is not None
    p.write_text("v2 -- same file, rewritten in place", encoding="utf-8")
    await asyncio.sleep(_TRAILING_EVENT_GRACE_S)
    row = await _row(rel)
    assert row is not None
    assert row.change_type != "deleted"
    assert row.status == "pending"
    assert row.mtime > first.mtime, "the in-place save was never delivered"


async def test_atomic_replace_over_existing_target_keeps_the_row_alive(
    runtime: MemoryRoot, watcher: CascadeWatcher
) -> None:
    """Write-temp-then-``os.replace`` is how many editors save.

    Windows reports it as REMOVED(target) + RENAMED(tmp -> target); FSEvents
    as a synthetic deletion of the old inode plus a move. In both, a deletion
    arrives for a path that still exists. Whatever the order, the target's row
    must end up live and the temp file must never have been enqueued.
    """
    target = _episode(runtime.root)
    target.write_text("v1", encoding="utf-8")
    rel = _rel(runtime.root, target)
    await _wait_row(rel)

    first = await _row(rel)
    assert first is not None
    tmp = target.with_name(target.name + ".tmp")  # not kind-matched
    tmp.write_text("v2 via atomic save", encoding="utf-8")
    os.replace(tmp, target)

    await asyncio.sleep(_TRAILING_EVENT_GRACE_S)
    row = await _row(rel)
    assert row is not None, f"target row vanished; rows: {await _all_paths()}"
    assert row.change_type != "deleted", (
        "an atomic save was recorded as a deletion -- the worker would wipe "
        "this file's LanceDB rows while the md is intact"
    )
    assert row.status == "pending"
    assert row.mtime > first.mtime, "the atomic save was never delivered"
    assert await _row(_rel(runtime.root, tmp)) is None
    assert target.read_text(encoding="utf-8") == "v2 via atomic save"


async def test_unlink_enqueues_deleted(
    runtime: MemoryRoot, watcher: CascadeWatcher
) -> None:
    p = _episode(runtime.root)
    p.write_text("doomed", encoding="utf-8")
    rel = _rel(runtime.root, p)
    await _wait_row(rel)
    p.unlink()
    row = await _wait_row(rel, lambda r: r.change_type == "deleted")
    assert row.mtime == 0.0


async def test_rename_within_root_moves_the_row(
    runtime: MemoryRoot, watcher: CascadeWatcher
) -> None:
    """The destination is the watcher's job; the source is the sweep's.

    ReadDirectoryChangesW reports a rename as RENAMED_OLD then RENAMED_NEW,
    and watchdog pairs them only when both land in the same read -- the
    pairing variable is local to one ``queue_events`` call. Split across two
    reads, the OLD leg is dropped and the watcher never learns the source
    path (CI saw the source row sit at ``added`` for 15 s). That is not a
    defect the watcher can fix; it is why the scanner exists: a state row
    whose path is gone from disk is re-emitted as ``deleted`` on the next
    sweep. So assert the immediate leg on the watcher and the source leg
    after one sweep, which is the contract the system actually offers on
    every backend.
    """
    a = _episode(runtime.root, "episode-2026-01-01.md")
    b = _episode(runtime.root, "episode-2026-01-02.md")
    a.write_text("renamed later", encoding="utf-8")
    rel_a, rel_b = _rel(runtime.root, a), _rel(runtime.root, b)
    await _wait_row(rel_a)
    os.rename(a, b)
    assert (await _wait_row(rel_b)).change_type in {"added", "modified"}
    await CascadeScanner(runtime).scan_once(kinds={"episode"})
    await _wait_row(rel_a, lambda r: r.change_type == "deleted", deadline=5)
