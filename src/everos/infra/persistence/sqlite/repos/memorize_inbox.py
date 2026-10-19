"""Repository for ``memorize_inbox`` — the async memorize queue.

Singleton bound to the process-wide ``sqlite_manager`` session factory.

Exposes the queue primitives the worker and the route need:

- :meth:`claim_next` — smallest-id ``queued`` row, flipped to ``processing``.
- :meth:`mark` — land a row in a new status (attempt / error bookkeeping).
- :meth:`count_pending` — cap checks (one session, or all).
- :meth:`requeue_processing` — startup crash recovery.
- :meth:`fail_abandoned` — the stuck-row sweep.
- :meth:`purge_expired` — retention sweep.
- :meth:`session_summary` — ``GET /status?session_id=``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import MemorizeInbox

PENDING_STATUSES: tuple[str, ...] = ("queued", "processing")


@dataclass(frozen=True)
class InboxSessionSummary:
    """Per-session queue snapshot for ``GET /status?session_id=``."""

    pending: int
    failed: int
    oldest_pending_created_at: dt.datetime | None


class _MemorizeInboxRepo(RepoBase[MemorizeInbox]):
    model = MemorizeInbox

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def claim_next(self) -> MemorizeInbox | None:
        """Flip the smallest-id ``queued`` row to ``processing`` and return it."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(MemorizeInbox)
                .where(MemorizeInbox.status == "queued")
                .order_by(MemorizeInbox.id.asc())  # type: ignore[union-attr]
                .limit(1)
            )
            row = (await s.execute(stmt)).scalars().first()
            if row is None:
                return None
            row.status = "processing"
            await s.commit()
            await s.refresh(row)
            return row

    async def mark(
        self,
        row_id: int,
        *,
        status: str,
        attempt: int | None = None,
        last_error: str | None = None,
        payload_json: str | None = None,
    ) -> None:
        """Set ``status`` (and optionally ``attempt`` / ``last_error`` /
        ``payload_json``) on one row. A delivered row passes ``payload_json=""``
        so the request body does not sit on disk for the retention period.
        """
        values: dict[str, object] = {"status": status}
        if attempt is not None:
            values["attempt"] = attempt
        if last_error is not None:
            values["last_error"] = last_error
        if payload_json is not None:
            values["payload_json"] = payload_json
        async with session_scope(self._factory) as s:
            await s.execute(
                update(MemorizeInbox).where(MemorizeInbox.id == row_id).values(**values)
            )
            await s.commit()

    async def count_pending(
        self,
        *,
        session_id: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
    ) -> int:
        """Rows still ``queued`` or ``processing``: one session's, or all."""
        where = [MemorizeInbox.status.in_(PENDING_STATUSES)]  # type: ignore[attr-defined]
        if session_id is not None:
            where += [
                MemorizeInbox.app_id == app_id,
                MemorizeInbox.project_id == project_id,
                MemorizeInbox.session_id == session_id,
            ]
        async with session_scope(self._factory) as s:
            stmt = select(func.count()).select_from(MemorizeInbox).where(*where)
            return int((await s.execute(stmt)).scalar_one())

    async def fail_abandoned(self) -> int:
        """Mark every ``processing`` row as ``failed``.

        Called only when no row is in flight: with the single worker idle,
        any ``processing`` row lost its status write and its outcome is
        unknown.
        """
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(MemorizeInbox)
                .where(MemorizeInbox.status == "processing")
                .values(
                    status="failed", last_error="status write lost; outcome unknown"
                )
            )
            await s.commit()
            return int(result.rowcount or 0)

    async def requeue_processing(self) -> int:
        """Return every ``processing`` row to ``queued`` (startup recovery).

        A row left ``processing`` was in flight when the process died; it is
        delivered again (at-least-once).
        """
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(MemorizeInbox)
                .where(MemorizeInbox.status == "processing")
                .values(status="queued")
            )
            await s.commit()
            return int(result.rowcount or 0)

    async def session_summary(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> InboxSessionSummary:
        scope = (
            MemorizeInbox.app_id == app_id,
            MemorizeInbox.project_id == project_id,
            MemorizeInbox.session_id == session_id,
        )
        pending_filter = MemorizeInbox.status.in_(PENDING_STATUSES)  # type: ignore[attr-defined]
        async with session_scope(self._factory) as s:
            pending = (
                await s.execute(
                    select(func.count())
                    .select_from(MemorizeInbox)
                    .where(*scope, pending_filter)
                )
            ).scalar_one()
            failed = (
                await s.execute(
                    select(func.count())
                    .select_from(MemorizeInbox)
                    .where(*scope, MemorizeInbox.status == "failed")
                )
            ).scalar_one()
            oldest = (
                await s.execute(
                    select(func.min(MemorizeInbox.created_at)).where(
                        *scope, pending_filter
                    )
                )
            ).scalar_one()
        return InboxSessionSummary(
            pending=int(pending),
            failed=int(failed),
            oldest_pending_created_at=oldest,
        )

    async def purge_expired(self, *, before: dt.datetime) -> tuple[int, int]:
        """Delete rows created before ``before`` unless still ``processing``.

        Two plain DELETEs rather than one with ``RETURNING`` (SQLite < 3.35
        has no ``RETURNING``). Only the worker changes ``queued`` rows and the
        sweep runs inside it, so the split is exact. Returns
        ``(deleted, deleted_queued)``; a non-zero second value means rows
        aged out without ever being delivered (the worker was down).
        """
        expired = MemorizeInbox.created_at < before
        async with session_scope(self._factory) as s:
            undelivered = await s.execute(
                delete(MemorizeInbox).where(expired, MemorizeInbox.status == "queued")
            )
            terminal = await s.execute(
                delete(MemorizeInbox).where(
                    expired,
                    MemorizeInbox.status.in_(("success", "failed")),  # type: ignore[attr-defined]
                )
            )
            await s.commit()
            queued = int(undelivered.rowcount or 0)
            return queued + int(terminal.rowcount or 0), queued


memorize_inbox_repo = _MemorizeInboxRepo()
