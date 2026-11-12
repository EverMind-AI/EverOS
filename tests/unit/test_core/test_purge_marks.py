"""Purge tombstones, the active job, and the work tracked while a purge waits."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from everos.core.purge_marks import MARKS, ORIGIN, PurgeScope, Tombstone
from everos.memory.ports import Event

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


def test_max_tombstone_wins(root: Path) -> None:
    marks = MARKS.for_root(root)
    later = datetime(2026, 11, 11, 11, 0, tzinfo=UTC)
    marks.set_tombstones([Tombstone(APP_A1, T10), Tombstone(PROJ_A1P1, later)])
    assert marks.tombstone_at("a1", "p1") == later
    assert marks.tombstone_at("a1", "p2") == T10
    assert marks.tombstone_at("a2", "p1") is None
    marks.remove_tombstone(Tombstone(PROJ_A1P1, later))
    assert marks.tombstone_at("a1", "p1") == T10


def test_track_counts_only_older_work_while_it_runs(root: Path) -> None:
    marks = MARKS.for_root(root)
    early, late = T10 - timedelta(seconds=1), T10 + timedelta(seconds=1)
    with marks.track(early):
        with marks.track(late):
            assert marks.tracked_before(T10, PROJ_A1P1) == 1
        with marks.track(early):
            assert marks.tracked_before(T10, PROJ_A1P1) == 2
        assert marks.tracked_before(T10, PROJ_A1P1) == 1
    assert marks.tracked_before(T10, PROJ_A1P1) == 0


def test_tracked_work_of_another_scope_is_not_counted(root: Path) -> None:
    """Work that names its scope counts only for a purge covering it; work
    that does not (an HTTP request) counts for every purge."""
    marks = MARKS.for_root(root)
    early = T10 - timedelta(seconds=1)
    with marks.track(early, ("a2", "p1")), marks.track(early, ("a1", "p1")):
        assert marks.tracked_before(T10, PROJ_A1P1) == 1
        assert marks.tracked_before(T10, APP_A1) == 1
        assert marks.tracked_before(T10, SPACE) == 2
    with marks.track(early):
        assert marks.tracked_before(T10, PROJ_A1P1) == 1


def test_track_ends_when_the_work_fails(root: Path) -> None:
    marks = MARKS.for_root(root)
    with pytest.raises(RuntimeError), marks.track(T10):
        raise RuntimeError("boom")
    assert marks.tracked_before(T10 + timedelta(1), SPACE) == 0


def test_roots_do_not_share_state(tmp_path: Path) -> None:
    one, two = (tmp_path / "one").resolve(), (tmp_path / "two").resolve()
    MARKS.for_root(one).raise_mark("j1", SPACE)
    MARKS.for_root(one).add_tombstone(Tombstone(SPACE, T10))
    assert MARKS.for_root(two).active_job_id is None
    assert MARKS.for_root(two).tombstone_at("a1", "p1") is None


def test_reset_clears_all_roots(root: Path) -> None:
    MARKS.for_root(root).raise_mark("j1", SPACE)
    MARKS.reset()
    assert MARKS.for_root(root).active_job_id is None


def test_event_ts_is_the_origin_inside_tracked_work() -> None:
    """An event made while handling an input carries that input's time."""
    token = ORIGIN.set(T10)
    try:
        assert Event().ts == T10
    finally:
        ORIGIN.reset(token)
    assert Event().ts > T10  # outside: when it is made
