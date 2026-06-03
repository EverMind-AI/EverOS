"""Repository for ``md_change_state`` — cascade work queue.

Sole writer of the table. The worker, watcher, scanner, and CLI all
go through this repo so the state-machine invariants (``processing``
claim semantics, retryable flag lifecycle) live in one place.

LSN ordering is **best-effort**, not strictly monotonic across
concurrent writers: :meth:`upsert` derives ``lsn = MAX(lsn) + 1``
which is a classic read-modify-write that two parallel writers could
race on (BEGIN DEFERRED leaves the SELECT half unprotected; cross-
process this is even more visible). The table schema does **not**
declare ``lsn UNIQUE`` and no caller depends on strict monotonicity —
the worker uses ``ORDER BY lsn LIMIT N`` for fairness only, and a
collision merely reorders two rows by a few ms; both rows are still
processed and the next upsert bumps the counter past the duplicate.
If a future feature needs strict monotonicity (e.g. CDC / audit log),
revisit by giving ``upsert`` its own ``BEGIN IMMEDIATE`` transaction.

Status values:

- ``pending`` — visible to the worker.
- ``processing`` — internal claim state (one worker is on it).
- ``done`` — handler succeeded.
- ``failed`` — handler exhausted retries or hit unrecoverable error
  (see ``retryable`` for the eligibility flag).
"""

from __future__ import annotations

import dataclasses

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.component.utils.datetime import get_utc_now
from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import MdChangeState


@dataclasses.dataclass(frozen=True)
class QueueSummary:
    """Aggregate counts for ``cascade status`` CLI output.

    ``pending`` includes the internal ``processing`` rows so the public
    state machine (12 doc §6) stays three-valued.
    """

    pending: int
    """Rows the worker hasn't completed yet (includes ``processing``)."""

    done: int
    """Rows landed successfully."""

    failed_retryable: int
    """``status='failed' AND retryable=TRUE`` — eligible for
    ``cascade fix --apply`` re-enqueue."""

    failed_permanent: int
    """``status='failed' AND retryable=FALSE`` — requires the user to
    edit the md and re-save."""

    max_lsn: int
    """Largest ``lsn`` ever assigned; 0 if the table is empty."""

    last_processed_lsn: int
    """Largest ``lsn`` whose row has reached a terminal state
    (``done`` or ``failed``); 0 if no terminal rows yet."""


class _MdChangeStateRepo(RepoBase[MdChangeState]):
    model = MdChangeState

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    # ── Writers: watcher / scanner / CLI sync ──────────────────────────────

    async def upsert(
        self,
        md_path: str,
        *,
        kind: str,
        change_type: str,
        mtime: float,
    ) -> int:
        """Enqueue or re-enqueue ``md_path``; return the assigned LSN.

        Behaviour:

        - **New row** → insert with ``status='pending'``,
          ``lsn = MAX(lsn) + 1``.
        - **Existing row** → bump ``last_changed_at``, refresh
          ``kind`` / ``change_type`` / ``mtime``, reset status back to
          ``pending``, zero ``retry_count`` / ``error`` / ``retryable``,
          and assign a fresh ``MAX(lsn) + 1`` so the worker re-processes
          this path *after* anything queued in between.

        The fresh LSN on re-enqueue is the property that lets the worker
        rely on ``ORDER BY lsn`` for ordering without losing fairness
        when a file flickers in and out of the queue. The ``MAX(lsn)+1``
        derivation is best-effort under concurrent writers — see module
        docstring for the trade-off.
        """
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            new_lsn = await _next_lsn(s)
            stmt = (
                sqlite_insert(MdChangeState)
                .values(
                    md_path=md_path,
                    kind=kind,
                    change_type=change_type,
                    mtime=mtime,
                    first_seen_at=now,
                    last_changed_at=now,
                    lsn=new_lsn,
                    status="pending",
                    retryable=None,
                    last_attempt_at=None,
                    retry_count=0,
                    error=None,
                )
                .on_conflict_do_update(
                    index_elements=["md_path"],
                    set_={
                        "kind": kind,
                        "change_type": change_type,
                        "mtime": mtime,
                        "last_changed_at": now,
                        "lsn": new_lsn,
                        "status": "pending",
                        "retryable": None,
                        "last_attempt_at": None,
                        "retry_count": 0,
                        "error": None,
                    },
                )
            )
            await s.execute(stmt)
            await s.commit()
            return new_lsn

    async def force_enqueue(self, md_path: str, kind: str) -> int:
        """`cascade sync --path` entry: re-enqueue regardless of status.

        Semantically the same as :meth:`upsert` with ``change_type
        ='modified'``; named separately because the CLI flow has no
        watcher / scanner event to attribute the row to.
        """
        return await self.upsert(
            md_path,
            kind=kind,
            change_type="modified",
            mtime=0.0,
        )

    # ── Worker claim ───────────────────────────────────────────────────────

    async def claim_one(self, md_path: str) -> MdChangeState | None:
        """Atomically transition one row ``pending → processing``.

        Implements the worker's claim contract: only the caller whose
        ``UPDATE`` returns ``rowcount == 1`` "owns" the row and should
        run the handler. All other concurrent callers get ``None`` and
        must move on (no exception — claim contention is not an error).
        """
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(MdChangeState)
                .where(MdChangeState.md_path == md_path)
                .where(MdChangeState.status == "pending")
                .values(status="processing", last_attempt_at=now)
            )
            await s.commit()
            if result.rowcount != 1:
                return None
            row = await s.get(MdChangeState, md_path)
            return row

    async def claim_pending_batch(self, limit: int = 100) -> list[MdChangeState]:
        """Claim up to ``limit`` pending rows in LSN order.

        Returns the claimed rows (now ``status='processing'``); empty
        list if none were pending. Sibling workers / processes may race
        on the same prefix — the per-row ``WHERE status='pending'``
        filter ensures each row lands in exactly one batch.
        """
        if limit <= 0:
            return []
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            picks = (
                (
                    await s.execute(
                        select(MdChangeState.md_path)
                        .where(MdChangeState.status == "pending")
                        .order_by(MdChangeState.lsn)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            if not picks:
                return []
            update_result = await s.execute(
                update(MdChangeState)
                .where(MdChangeState.md_path.in_(picks))
                .where(MdChangeState.status == "pending")
                .values(status="processing", last_attempt_at=now)
            )
            await s.commit()
            if update_result.rowcount == 0:
                return []
            rows = (
                (
                    await s.execute(
                        select(MdChangeState)
                        .where(MdChangeState.md_path.in_(picks))
                        .where(MdChangeState.status == "processing")
                        .order_by(MdChangeState.lsn)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    # ── Worker result reporting ────────────────────────────────────────────

    async def mark_done(self, md_path: str) -> None:
        """Transition the row to ``done`` after a successful handler run.

        Guarded by ``WHERE status='processing'`` so the call is a no-op
        if a concurrent :meth:`upsert` (watcher / scanner re-enqueue)
        has flipped the row back to ``pending`` while the worker was
        running the handler. In that case the next
        :meth:`claim_pending_batch` drain re-runs the handler against
        the latest md state — losing the stale ``done`` write rather
        than the new ``pending`` is the correct trade.
        """
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            await s.execute(
                update(MdChangeState)
                .where(MdChangeState.md_path == md_path)
                .where(MdChangeState.status == "processing")
                .values(
                    status="done",
                    last_attempt_at=now,
                    error=None,
                    retryable=None,
                )
            )
            await s.commit()

    async def mark_failed(
        self,
        md_path: str,
        *,
        retryable: bool,
        error: str,
        new_retry_count: int,
    ) -> None:
        """Transition the row to ``failed`` with the given diagnostic.

        Args:
            md_path: The row's primary key.
            retryable: ``True`` for transient failures (HTTP 5xx,
                connection reset, 429) — ``cascade fix --apply`` will
                re-enqueue. ``False`` for unrecoverable failures
                (YAML parse, schema mismatch) — needs user edit.
            error: Truncated failure message for ``cascade fix`` output.
            new_retry_count: The retry count *after* this attempt (the
                caller knows whether it was a retry or the final
                attempt).

        Guarded by ``WHERE status='processing'`` for the same reason as
        :meth:`mark_done` — a concurrent re-enqueue must win over a
        terminal write tied to a stale claim.
        """
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            # Same guard as ``mark_done``: only flip ``processing → failed``.
            # A concurrent watcher / scanner upsert may have reset the row
            # back to ``pending`` (file changed during processing) — in
            # that case the failure verdict is stale and we let the next
            # drain re-attempt against the new md state instead of
            # stamping ``failed`` over the live pending row.
            await s.execute(
                update(MdChangeState)
                .where(MdChangeState.md_path == md_path)
                .where(MdChangeState.status == "processing")
                .values(
                    status="failed",
                    retryable=retryable,
                    last_attempt_at=now,
                    error=error,
                    retry_count=new_retry_count,
                )
            )
            await s.commit()

    # ── Startup recovery ───────────────────────────────────────────────────

    async def recover_orphan_processing(self) -> int:
        """Reset every ``processing`` row to ``pending``; return the count.

        Cascade runs single-process today, so any row in ``processing``
        when the orchestrator boots is leftover from a prior crash
        (the worker died between :meth:`claim_pending_batch` and
        ``mark_done`` / ``mark_failed``). Idempotent — no rows in
        ``processing`` is a clean no-op.
        """
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(MdChangeState)
                .where(MdChangeState.status == "processing")
                .values(status="pending", last_attempt_at=None)
            )
            await s.commit()
            return int(result.rowcount or 0)

    # ── CLI fix / status ───────────────────────────────────────────────────

    async def list_failed(self) -> list[MdChangeState]:
        """Return every ``status='failed'`` row, oldest LSN first.

        Drives the ``cascade fix`` (no ``--apply``) preview table — the
        CLI splits the result by ``retryable`` into two sections.
        """
        async with session_scope(self._factory) as s:
            rows = (
                (
                    await s.execute(
                        select(MdChangeState)
                        .where(MdChangeState.status == "failed")
                        .order_by(MdChangeState.lsn)
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    async def reset_retryable_to_pending(self) -> int:
        """`cascade fix --apply` engine: re-enqueue every retryable row.

        Affects only ``status='failed' AND retryable=TRUE``. Rows with
        ``retryable=FALSE`` are left untouched — they need the user to
        edit the md and re-save (the scanner / watcher will pick up the
        change and re-enqueue them naturally).

        Returns the number of rows transitioned.
        """
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(MdChangeState)
                .where(MdChangeState.status == "failed")
                .where(MdChangeState.retryable.is_(True))
                .values(
                    status="pending",
                    retryable=None,
                    retry_count=0,
                    error=None,
                    last_changed_at=now,
                )
            )
            await s.commit()
            return int(result.rowcount or 0)

    async def queue_summary(self) -> QueueSummary:
        """Aggregate the table for the ``cascade status`` CLI."""
        async with session_scope(self._factory) as s:
            pending = await _count_where(
                s, MdChangeState.status.in_(["pending", "processing"])
            )
            done = await _count_where(s, MdChangeState.status == "done")
            failed_retryable = await _count_where(
                s,
                (MdChangeState.status == "failed")
                & (MdChangeState.retryable.is_(True)),
            )
            failed_permanent = await _count_where(
                s,
                (MdChangeState.status == "failed")
                & (MdChangeState.retryable.is_(False)),
            )
            max_lsn_stmt = select(func.coalesce(func.max(MdChangeState.lsn), 0))
            max_lsn = int((await s.execute(max_lsn_stmt)).scalar_one())
            last_processed_lsn = int(
                (
                    await s.execute(
                        select(func.coalesce(func.max(MdChangeState.lsn), 0)).where(
                            MdChangeState.status.in_(["done", "failed"])
                        )
                    )
                ).scalar_one()
            )
        return QueueSummary(
            pending=pending,
            done=done,
            failed_retryable=failed_retryable,
            failed_permanent=failed_permanent,
            max_lsn=max_lsn,
            last_processed_lsn=last_processed_lsn,
        )


async def _next_lsn(session: AsyncSession) -> int:
    """Pick the next global LSN (``MAX(lsn) + 1``).

    Called inside the same write transaction as the UPSERT so SQLite's
    WAL writer serialisation guarantees no two writers see the same
    ``MAX``. Empty table returns 1.
    """
    result = await session.execute(
        select(func.coalesce(func.max(MdChangeState.lsn), 0))
    )
    return int(result.scalar_one()) + 1


async def _count_where(session: AsyncSession, predicate: object) -> int:
    """``SELECT COUNT(*) WHERE <predicate>`` returning a Python int."""
    stmt = select(func.count()).select_from(MdChangeState).where(predicate)  # type: ignore[arg-type]
    return int((await session.execute(stmt)).scalar_one())


md_change_state_repo = _MdChangeStateRepo()
