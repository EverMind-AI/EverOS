"""The purge gate's verdict for every kind of event and purge state."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from everos.core.purge_marks import (
    CURRENT_EVENT,
    MARKS,
    EventStamp,
    PurgeScope,
    Tombstone,
)
from everos.memory.ports import CronTick, Event
from everos.service.purge import PurgeRunGate

T = datetime(2026, 11, 11, 10, 0, tzinfo=UTC)
BEFORE, AFTER = T - timedelta(minutes=1), T + timedelta(minutes=1)
S = PurgeScope("project", "a1", "p1", "a1", "p1")


class _Scoped(Event):
    app_id: str = "a1"
    project_id: str = "p1"


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()


# (event, tombstone of S at T?, mark up?) -> verdict, written out by hand.
TABLE = [
    (_Scoped(ts=BEFORE), False, False, "run"),
    (_Scoped(ts=BEFORE), False, True, "wait"),
    (_Scoped(ts=BEFORE), True, False, "discard"),
    (_Scoped(ts=BEFORE), True, True, "discard"),
    (_Scoped(ts=AFTER), True, False, "run"),
    (_Scoped(ts=AFTER), True, True, "wait"),
    (_Scoped(ts=BEFORE, project_id="p2"), True, False, "run"),
    (_Scoped(ts=BEFORE, project_id="p2"), True, True, "wait"),
    (CronTick(strategy_name="s", ts=BEFORE), True, False, "run"),
    (CronTick(strategy_name="s", ts=BEFORE), True, True, "wait"),
]


@pytest.mark.parametrize(("event", "tombstone", "marked", "expected"), TABLE)
def test_gate_verdict_table(
    tmp_path: Path, event: Event, tombstone: bool, marked: bool, expected: str
) -> None:
    marks = MARKS.for_root(tmp_path)
    if tombstone:
        marks.add_tombstone(Tombstone(S, T))
    if marked:
        marks.raise_mark("j1", PurgeScope("project", "a9", "p9", "a9", "p9"))
    assert PurgeRunGate(marks).verdict(event) == expected


def test_bind_stamps_scoped_events_only(tmp_path: Path) -> None:
    gate = PurgeRunGate(MARKS.for_root(tmp_path))
    with gate.bind(_Scoped(ts=AFTER)):
        assert CURRENT_EVENT.get() == EventStamp("a1", "p1", AFTER)
    assert CURRENT_EVENT.get() is None
    with gate.bind(CronTick(strategy_name="s", ts=AFTER)):
        assert CURRENT_EVENT.get() is None
