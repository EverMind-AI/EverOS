"""RunRecord persistence — state machine writes + completed-attempt history.

State machine (one row per ``run_id``):
    RUNNING  →  SUCCESS / FAILED / DEAD_LETTER / CRASHED

RUNNING rows are recovery state and never count toward the history limit.
Every status update atomically keeps the most recently finished
``max_records_per_strategy`` attempts for that strategy, including FAILED.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from everos.core.utils.datetime import (
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
        event_id: str,
    ) -> None:
        """Insert a recoverable RUNNING row, independent of the history limit."""
        async with self._storage.write() as conn:
            await conn.execute(
                "INSERT INTO run_record "
                "(run_id, strategy_name, status, attempt, started_at, "
                " event_topic, event_payload, max_retries_snapshot, event_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    strategy_name,
                    RunStatus.RUNNING.value,
                    attempt,
                    to_iso_format(get_utc_now()),
                    event_topic,
                    event_payload,
                    max_retries_snapshot,
                    event_id,
                ),
            )

    async def mark_success(self, *, run_id: str, finished_at: datetime) -> None:
        """Mark RUNNING → SUCCESS, dropping the event's text.

        Only a RUNNING row is ever replayed (crash recovery), so a finished
        run's copy of the conversation is kept nowhere it is needed -- and it
        would outlive a deletion of that conversation. Failed and dead-letter
        rows keep theirs for diagnosis, rolled off with the per-strategy cap.
        """
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
        payload = ", event_payload = ''" if status is RunStatus.SUCCESS else ""
        async with self._storage.transaction() as conn:
            await conn.execute(
                "UPDATE run_record "
                f"SET status = ?, finished_at = ?, error = ?{payload} "
                "WHERE run_id = ?",
                (status.value, to_iso_format(finished_at), error, run_id),
            )
            await conn.execute(
                "DELETE FROM run_record "
                "WHERE status != ? AND strategy_name = ("
                "  SELECT strategy_name FROM run_record WHERE run_id = ?) "
                "AND run_id NOT IN ("
                "  SELECT run_id FROM run_record "
                "  WHERE status != ? AND strategy_name = ("
                "    SELECT strategy_name FROM run_record WHERE run_id = ?) "
                "  ORDER BY finished_at DESC, run_id DESC LIMIT ?)",
                (
                    RunStatus.RUNNING.value,
                    run_id,
                    RunStatus.RUNNING.value,
                    run_id,
                    self._max,
                ),
            )

    async def redact_payloads(self, match: Callable[[str], bool], marker: str) -> int:
        """Replace the payload of finished, unsuccessful rows ``match`` accepts.

        A SUCCESS row's payload is already empty and a RUNNING row is live
        recovery state; every other row must keep a non-empty payload, so it
        gets ``marker`` instead of the original. Returns rows rewritten.
        """
        async with self._storage.transaction() as conn:
            cur = await conn.execute(
                "SELECT run_id, event_payload FROM run_record "
                "WHERE status NOT IN (?, ?) AND event_payload != ?",
                (RunStatus.RUNNING.value, RunStatus.SUCCESS.value, marker),
            )
            hits = [
                run_id for run_id, payload in await cur.fetchall() if match(payload)
            ]
            for run_id in hits:
                await conn.execute(
                    "UPDATE run_record SET event_payload = ? WHERE run_id = ?",
                    (marker, run_id),
                )
        return len(hits)

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

    async def list_by_event_id(self, event_id: str) -> list[RunRecord]:
        """Return all runs triggered by the given ``event_id``."""
        async with self._storage.connect() as conn:
            cur = await conn.execute(
                _SELECT_COLUMNS + " WHERE event_id = ? ORDER BY started_at DESC",
                (event_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]


_SELECT_COLUMNS = (
    "SELECT run_id, strategy_name, status, attempt, started_at, finished_at, "
    "       error, event_topic, event_payload, max_retries_snapshot, event_id "
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
        event_id=row[10],
    )
