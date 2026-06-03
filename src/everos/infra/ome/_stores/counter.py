"""CounterStore — persistent (strategy_name, bucket_key) → counter rows.

Backs the ``Counter`` gate in OME's dispatch pipeline: each call to
:meth:`CounterStore.incr_and_check` atomically increments the bucket's
counter and reports whether the strategy should fire this time.

Pass semantics:
  - ``counter >= threshold`` AND cooldown elapsed → ``passed=True``
  - On pass, the row's counter resets to 0 and ``last_passed_ts``
    advances to ``now``; the next pass needs a fresh accumulation.
  - ``cooldown_seconds=0`` disables the cooldown gate (threshold alone).
"""

from __future__ import annotations

from datetime import timedelta

from everos.component.utils.datetime import (
    from_iso_format,
    get_utc_now,
    to_iso_format,
)
from everos.infra.ome._stores.storage import OMEStorage


class CounterStore:
    """SQLite-backed counter for the ``Counter`` gate (see module docstring)."""

    def __init__(self, storage: OMEStorage) -> None:
        self._storage = storage

    async def incr_and_check(
        self,
        strategy_name: str,
        bucket_key: str,
        *,
        threshold: int,
        cooldown_seconds: int,
    ) -> tuple[bool, int]:
        """Increment ``(strategy_name, bucket_key)``'s counter atomically.

        Args:
            strategy_name: Strategy whose counter to update.
            bucket_key: The bucket value derived from the event field
                (or ``"__all__"`` when the gate is unbucketed).
            threshold: Pass once the counter reaches this value
                (``>=``).
            cooldown_seconds: Minimum seconds since the last pass for
                the strategy/bucket; ``0`` disables the cooldown check.

        Returns:
            ``(passed, counter)``. ``counter`` is the counter value at
            the moment of the check (i.e. pre-reset on pass). Useful for
            diagnostics — ``threshold`` is *not* substituted, so callers
            observing ``counter > threshold`` learn the gate is
            over-armed (e.g. threshold was lowered via hot reload while
            the counter had already accumulated past the new value).
        """
        now = get_utc_now()
        async with self._storage.transaction() as conn:
            cur = await conn.execute(
                "SELECT counter, last_passed_ts FROM counter_store "
                "WHERE strategy_name = ? AND bucket_key = ?",
                (strategy_name, bucket_key),
            )
            row = await cur.fetchone()
            counter = (row[0] if row else 0) + 1
            last_passed = from_iso_format(row[1]) if row and row[1] else None

            cooldown_ok = (
                cooldown_seconds == 0
                or last_passed is None
                or now - last_passed >= timedelta(seconds=cooldown_seconds)
            )
            passed = counter >= threshold and cooldown_ok

            new_counter = 0 if passed else counter
            new_last_passed_ts = (
                to_iso_format(now)
                if passed
                else (to_iso_format(last_passed) if last_passed else None)
            )
            await conn.execute(
                "INSERT INTO counter_store (strategy_name, bucket_key, "
                "counter, last_passed_ts) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(strategy_name, bucket_key) DO UPDATE SET "
                "counter = excluded.counter, "
                "last_passed_ts = excluded.last_passed_ts",
                (strategy_name, bucket_key, new_counter, new_last_passed_ts),
            )
            return passed, counter

    async def get_progress(self, strategy_name: str, bucket_key: str) -> int:
        """Return the counter value persisted for this bucket (0 if absent).

        Read-only; does not increment. Used by dispatcher inspect-mode
        to report progress without mutating state.
        """
        async with self._storage.connect() as conn:
            cur = await conn.execute(
                "SELECT counter FROM counter_store "
                "WHERE strategy_name = ? AND bucket_key = ?",
                (strategy_name, bucket_key),
            )
            row = await cur.fetchone()
            return row[0] if row else 0
