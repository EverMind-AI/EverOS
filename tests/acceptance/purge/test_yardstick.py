"""A37: the "range cleared" yardstick every case uses (``observe.s_leftovers``).

Its ome.db part, which no purge case can make fail on demand: a run row
still carrying the content of an event of the range, older than the purge,
is left over; the same event newer than the purge, another range's event
and the purge's marker are not.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from .observe import s_leftovers

pytestmark = pytest.mark.acceptance

T = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)


def _event(app: str, project: str, ts: datetime) -> str:
    return json.dumps({"app_id": app, "project_id": project, "ts": ts.isoformat()})


def test_only_an_old_event_of_the_range_left_in_ome_db_counts(tmp_path: Path) -> None:
    db = tmp_path / ".index" / "sqlite" / "ome.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE run_record (strategy_name TEXT, status TEXT, event_payload TEXT)"
    )
    second = timedelta(seconds=1)
    con.executemany(
        "INSERT INTO run_record VALUES (?, ?, ?)",
        [
            ("old_s", "failed", _event("a1", "p1", T - second)),
            ("new_s", "failed", _event("a1", "p1", T + second)),
            ("old_other", "failed", _event("a2", "p1", T - second)),
            ("marker", "dead_letter", json.dumps({"purged_by": "pj_1"})),
        ],
    )
    con.commit()
    con.close()
    assert s_leftovers(tmp_path, "a1", "p1", before=T) == {
        "ome.run_record_payload": [("old_s", "failed")]
    }
