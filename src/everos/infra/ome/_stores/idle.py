"""IdleStore — last_activity_ts rows backing the Idle trigger.

All writes pass through ``to_iso_format`` over a tz-aware datetime, so
``last_activity_ts`` is a fixed-format ISO 8601 string whose
lexicographic order matches temporal order — :meth:`scan_idle` relies
on this to keep the column un-wrapped in its predicate so SQLite can
use ``idx_idle_scan``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from everos.component.utils.datetime import from_iso_format, to_iso_format
from everos.infra.ome._stores.storage import OMEStorage


class IdleStore:
    """SQLite-backed last-activity tracker for the ``Idle`` trigger."""

    def __init__(self, storage: OMEStorage) -> None:
        self._storage = storage

    async def touch(self, strategy_name: str, bucket_key: str, *, at: datetime) -> None:
        """UPSERT ``last_activity_ts = at`` for ``(strategy_name, bucket_key)``."""
        async with self._storage.connect() as conn:
            await conn.execute(
                "INSERT INTO idle_store "
                "(strategy_name, bucket_key, last_activity_ts) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(strategy_name, bucket_key) DO UPDATE SET "
                "last_activity_ts = excluded.last_activity_ts",
                (strategy_name, bucket_key, to_iso_format(at)),
            )
            await conn.commit()

    async def scan_idle(
        self, strategy_name: str, *, idle_seconds: int, now: datetime
    ) -> list[str]:
        """Return bucket_keys with ``last_activity_ts`` older than ``idle_seconds``."""
        # Cutoff on the RHS so the indexed column stays un-wrapped.
        cutoff = to_iso_format(now - timedelta(seconds=idle_seconds))
        async with self._storage.connect() as conn:
            cur = await conn.execute(
                "SELECT bucket_key FROM idle_store "
                "WHERE strategy_name = ? AND last_activity_ts <= ? "
                "ORDER BY last_activity_ts ASC",
                (strategy_name, cutoff),
            )
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def get_last_activity(
        self, strategy_name: str, bucket_key: str
    ) -> datetime | None:
        """Return the stored ``last_activity_ts`` (``None`` if never touched)."""
        async with self._storage.connect() as conn:
            cur = await conn.execute(
                "SELECT last_activity_ts FROM idle_store "
                "WHERE strategy_name = ? AND bucket_key = ?",
                (strategy_name, bucket_key),
            )
            row = await cur.fetchone()
        return from_iso_format(row[0]) if row else None
