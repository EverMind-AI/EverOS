"""RunRecord persistence — state machine writes + same-transaction ring-buffer trim.

State machine (one row per ``run_id``):
    RUNNING  →  SUCCESS / FAILED / DEAD_LETTER / CRASHED

Every :meth:`RunRecordStore.mark_running` INSERT runs inside one
``BEGIN IMMEDIATE`` transaction with a paired DELETE that keeps only
the newest ``max_records_per_strategy`` rows for that strategy. Bound
is enforced atomically — no background sweeper, no transient
over-budget state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from everos.component.utils.datetime import (
    from_iso_format,
    get_utc_now,
    to_iso_format,
)
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.records import RunRecord, RunStatus


class RunRecordStore:
    """SQLite-backed persistence for ``RunRecord`` (see module docstring)."""

    def __init__(self, storage: OMEStorage, max_records_per_strategy: int) -> None:
        self._storage = storage
        self._max = max_records_per_strategy

    async def mark_running(
        self,
        *,
        run_id: str,
        strategy_name: str,
        attempt: int,
        event_topic: str,
        event_payload: str,
        max_retries_snapshot: int,
    ) -> None:
        """Insert a new RUNNING row and trim the strategy's ring buffer atomically."""
        async with self._storage.transaction() as conn:
            await conn.execute(
                "INSERT INTO run_record "
                "(run_id, strategy_name, status, attempt, started_at, "
                " event_topic, event_payload, max_retries_snapshot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    strategy_name,
                    RunStatus.RUNNING.value,
                    attempt,
                    to_iso_format(get_utc_now()),
                    event_topic,
                    event_payload,
                    max_retries_snapshot,
                ),
            )
            await conn.execute(
                "DELETE FROM run_record "
                "WHERE strategy_name = ? AND run_id NOT IN ("
                "  SELECT run_id FROM run_record WHERE strategy_name = ? "
                "  ORDER BY started_at DESC LIMIT ?)",
                (strategy_name, strategy_name, self._max),
            )

    async def mark_success(self, *, run_id: str, finished_at: datetime) -> None:
        """Mark RUNNING → SUCCESS."""
        await self._update_status(run_id, RunStatus.SUCCESS, finished_at, None)

    async def mark_failed(
        self, *, run_id: str, finished_at: datetime, error: str
    ) -> None:
        """Mark RUNNING → FAILED (retry pending)."""
        await self._update_status(run_id, RunStatus.FAILED, finished_at, error)

    async def mark_dead_letter(
        self, *, run_id: str, finished_at: datetime, error: str
    ) -> None:
        """Mark RUNNING → DEAD_LETTER (retries exhausted or non-retryable)."""
        await self._update_status(run_id, RunStatus.DEAD_LETTER, finished_at, error)

    async def mark_crashed(
        self, *, run_id: str, finished_at: datetime, error: str
    ) -> None:
        """Mark RUNNING → CRASHED (called by crash-recovery sweep)."""
        await self._update_status(run_id, RunStatus.CRASHED, finished_at, error)

    async def _update_status(
        self,
        run_id: str,
        status: RunStatus,
        finished_at: datetime,
        error: str | None,
    ) -> None:
        async with self._storage.connect() as conn:
            await conn.execute(
                "UPDATE run_record "
                "SET status = ?, finished_at = ?, error = ? "
                "WHERE run_id = ?",
                (status.value, to_iso_format(finished_at), error, run_id),
            )
            await conn.commit()

    async def get(self, run_id: str) -> RunRecord | None:
        """Return the record for ``run_id`` (``None`` if absent)."""
        async with self._storage.connect() as conn:
            cur = await conn.execute(
                _SELECT_COLUMNS + " WHERE run_id = ?",
                (run_id,),
            )
            row = await cur.fetchone()
        return _row_to_record(row) if row else None

    async def list_runs(
        self,
        *,
        strategy_name: str,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        """Return ``strategy_name``'s records, newest first; optional status filter."""
        sql = _SELECT_COLUMNS + " WHERE strategy_name = ?"
        args: list[Any] = [strategy_name]
        if status is not None:
            sql += " AND status = ?"
            args.append(status.value)
        sql += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        async with self._storage.connect() as conn:
            cur = await conn.execute(sql, args)
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]

    async def find_running(self) -> list[RunRecord]:
        """Return every row still in RUNNING — used by crash recovery at start()."""
        async with self._storage.connect() as conn:
            cur = await conn.execute(
                _SELECT_COLUMNS + " WHERE status = ?",
                (RunStatus.RUNNING.value,),
            )
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]


_SELECT_COLUMNS = (
    "SELECT run_id, strategy_name, status, attempt, started_at, finished_at, "
    "       error, event_topic, event_payload, max_retries_snapshot "
    "FROM run_record"
)


def _row_to_record(row: tuple) -> RunRecord:
    return RunRecord(
        run_id=row[0],
        strategy_name=row[1],
        status=RunStatus(row[2]),
        attempt=row[3],
        started_at=from_iso_format(row[4]),
        finished_at=from_iso_format(row[5]) if row[5] else None,
        error=row[6],
        event_topic=row[7],
        event_payload=row[8],
        max_retries_snapshot=row[9],
    )
