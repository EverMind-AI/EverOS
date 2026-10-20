"""One-off backfill for the ``memorized_message`` local patch.

Records the timestamp of every message already inside a memcell, so the first
re-post after deploying the patch is deduped too. Run with the daemon stopped
(it creates the table if missing; re-running is harmless):

    .venv/bin/python scripts/backfill_memorized_messages.py <path/to/system.db>
"""

from __future__ import annotations

import json
import sqlite3
import sys

from sqlalchemy import create_engine

from everos.infra.persistence.sqlite.tables import MemorizedMessage


def main(db_path: str) -> None:
    MemorizedMessage.__table__.create(  # type: ignore[attr-defined]
        create_engine(f"sqlite:///{db_path}"), checkfirst=True
    )
    conn = sqlite3.connect(db_path)
    before = conn.execute("SELECT count(*) FROM memorized_message").fetchone()[0]
    cells = 0
    rows: set[tuple[str, str, str, int]] = set()
    for app_id, project_id, session_id, payload in conn.execute(
        "SELECT app_id, project_id, session_id, payload_json FROM memcell"
    ):
        cells += 1
        for item in json.loads(payload).get("items", []):
            rows.add((app_id, project_id, session_id, int(item["timestamp"])))
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO memorized_message "
            "(app_id, project_id, session_id, timestamp_ms) VALUES (?, ?, ?, ?)",
            rows,
        )
    after = conn.execute("SELECT count(*) FROM memorized_message").fetchone()[0]
    print(f"memcells={cells} timestamps={len(rows)} rows {before} -> {after}")


if __name__ == "__main__":
    main(sys.argv[1])
