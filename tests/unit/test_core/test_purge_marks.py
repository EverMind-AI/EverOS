"""The purge mark and tombstones: which writes a purge refuses, and when."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from everos.core.errors import PurgeInProgressError
from everos.core.purge_marks import (
    CURRENT_EVENT,
    MARKS,
    PURGING,
    EventStamp,
    PurgeScope,
    Tombstone,
    ensure_writable,
)

T10 = datetime(2026, 11, 11, 10, 0, tzinfo=UTC)
SPACE = PurgeScope("space")
APP_A1 = PurgeScope("app", app_id="a1", app_dir="a1")
PROJ_A1P1 = PurgeScope(
    "project", app_id="a1", project_id="p1", app_dir="a1", project_dir="p1"
)


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path.resolve()


# (scope, (app, project)) -> covered?  Written out by hand.
COVERS = [
    (SPACE, ("a1", "p1"), True),
    (SPACE, ("a1", "p2"), True),
    (SPACE, ("a2", "p1"), True),
    (APP_A1, ("a1", "p1"), True),
    (APP_A1, ("a1", "p2"), True),
    (APP_A1, ("a2", "p1"), False),
    (PROJ_A1P1, ("a1", "p1"), True),
    (PROJ_A1P1, ("a1", "p2"), False),
    (PROJ_A1P1, ("a2", "p1"), False),
]


@pytest.mark.parametrize(("scope", "where", "expected"), COVERS)
def test_covers_matrix(
    scope: PurgeScope, where: tuple[str, str], expected: bool
) -> None:
    assert scope.covers(*where) is expected
    assert scope.covers_dirs(*where) is expected


def test_ensure_writable_blocks_target_scope_only(root: Path) -> None:
    MARKS.for_root(root).raise_mark("j1", PROJ_A1P1)
    with pytest.raises(PurgeInProgressError, match="j1"):
        ensure_writable(root / "a1" / "p1" / "users" / "u1" / "user.md")
    ensure_writable(root / "a1" / "p2" / "users" / "u1" / "user.md")
    ensure_writable(root / "a2" / "p1" / "users" / "u1" / "user.md")
    ensure_writable(root.parent / "elsewhere" / "a1" / "p1" / "x.md")


def test_purging_task_passes(root: Path) -> None:
    MARKS.for_root(root).raise_mark("j1", SPACE)
    token = PURGING.set(True)
    try:
        ensure_writable(root / "a1" / "p1" / "x.md")
    finally:
        PURGING.reset(token)
    with pytest.raises(PurgeInProgressError):
        ensure_writable(root / "a1" / "p1" / "x.md")


@pytest.mark.parametrize(
    ("event_at", "rejected"),
    [
        (datetime(2026, 11, 11, 9, 59, tzinfo=UTC), True),
        (datetime(2026, 11, 11, 10, 1, tzinfo=UTC), False),
    ],
)
def test_stale_event_rejected_after_lift(
    root: Path, event_at: datetime, rejected: bool
) -> None:
    marks = MARKS.for_root(root)
    marks.add_tombstone(Tombstone(PROJ_A1P1, T10))  # the mark is already down
    token = CURRENT_EVENT.set(EventStamp("a1", "p1", event_at))
    try:
        if rejected:
            with pytest.raises(PurgeInProgressError):
                ensure_writable(root / "a1" / "p1" / "x.md")
        else:
            ensure_writable(root / "a1" / "p1" / "x.md")
        ensure_writable(root / "a1" / "p2" / "x.md")  # other scope: never stale
    finally:
        CURRENT_EVENT.reset(token)
    ensure_writable(root / "a1" / "p1" / "x.md")  # not a background event


@pytest.mark.parametrize(
    "rel", [".tmp/purge-j1/a1/p1/x.md", ".index/sqlite/system.db", "ome.toml"]
)
def test_system_dirs_are_not_scopes(root: Path, rel: str) -> None:
    MARKS.for_root(root).raise_mark("j1", SPACE)
    ensure_writable(root / rel)


def test_dot_named_app_is_a_scope(root: Path) -> None:
    MARKS.for_root(root).raise_mark("j1", PurgeScope("app", ".foo", None, ".foo"))
    with pytest.raises(PurgeInProgressError):
        ensure_writable(root / ".foo" / "p1" / "x.md")


def test_max_tombstone_wins(root: Path) -> None:
    marks = MARKS.for_root(root)
    later = datetime(2026, 11, 11, 11, 0, tzinfo=UTC)
    marks.set_tombstones([Tombstone(APP_A1, T10), Tombstone(PROJ_A1P1, later)])
    assert marks.tombstone_at("a1", "p1") == later
    assert marks.tombstone_at("a1", "p2") == T10
    assert marks.tombstone_at("a2", "p1") is None
    assert marks.tombstone_at_dirs("a1", "p1") == later
    marks.remove_tombstone(Tombstone(PROJ_A1P1, later))
    assert marks.tombstone_at("a1", "p1") == T10


def test_roots_do_not_share_marks(tmp_path: Path) -> None:
    one, two = (tmp_path / "one").resolve(), (tmp_path / "two").resolve()
    MARKS.for_root(one).raise_mark("j1", SPACE)
    assert MARKS.for_root(two).active_job_id is None
    ensure_writable(two / "a1" / "p1" / "x.md")
    with pytest.raises(PurgeInProgressError):
        ensure_writable(one / "a1" / "p1" / "x.md")


def test_reset_clears_all_roots(root: Path) -> None:
    MARKS.for_root(root).raise_mark("j1", SPACE)
    MARKS.reset()
    assert MARKS.for_root(root).active_job_id is None


async def test_wait_lifted_returns_when_mark_lifts(root: Path) -> None:
    marks = MARKS.for_root(root)
    await asyncio.wait_for(marks.wait_lifted(), 1)  # no mark: at once
    marks.raise_mark("j1", SPACE)
    waiter = asyncio.create_task(marks.wait_lifted())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    marks.lift_mark()
    await asyncio.wait_for(waiter, 1)
    assert marks.active_job_id is None and marks.active_scope is None
