"""Repository for ``memorize_inbox`` — the async memorize queue.

Singleton bound to the process-wide ``sqlite_manager`` session factory.

Exposes the queue primitives the worker and the route need:

- :meth:`claim_next` — smallest-id ``queued`` row outside the sessions in
  flight, flipped to ``processing``.
- :meth:`mark` — land a row in a new status (attempt / error bookkeeping).
- :meth:`count_pending` — cap checks (one session, or all).
- :meth:`requeue_processing` — startup crash recovery.
- :meth:`fail_abandoned` — the stuck-row sweep.
- :meth:`purge_expired` — retention sweep.
- :meth:`list_tasks` — ``GET /tasks`` (filter by session / status, paged).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import MemorizeInbox

PENDING_STATUSES: tuple[str, ...] = ("queued", "processing")
# ``last_error`` of a row the sweep failed because its status write was lost.
LOST_STATUS_WRITE = "status write lost; outcome unknown"
# ``last_error`` prefix of a row start() gave up on after too many replays.
REPLAYS_EXHAUSTED_PREFIX = "replayed "


def replays_exhausted(n: int) -> str:
    return f"{REPLAYS_EXHAUSTED_PREFIX}{n} times without finishing; giving up"


# Joins (app_id, project_id, session_id) into one comparable key in SQL.
_SCOPE_SEP = "\x1f"


class _MemorizeInboxRepo(RepoBase[MemorizeInbox]):
    model = MemorizeInbox

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def claim_next(
        self, *, exclude_sessions: Collection[tuple[str, str, str]] = ()
    ) -> MemorizeInbox | None:
        """Flip the smallest-id ``queued`` row to ``processing`` and return it.

        ``exclude_sessions`` are ``(app_id, project_id, session_id)`` scopes
        that already have a row in flight; their queued rows are skipped so a
        session never runs two batches at once and keeps its arrival order.
        """
        async with session_scope(self._factory) as s:
            stmt = (
                select(MemorizeInbox)
                .where(MemorizeInbox.status == "queued")
                .order_by(MemorizeInbox.id.asc())  # type: ignore[union-attr]
                .limit(1)
            )
            if exclude_sessions:
                scope_key = (
                    MemorizeInbox.app_id  # type: ignore[operator]
                    + _SCOPE_SEP
                    + MemorizeInbox.project_id
                    + _SCOPE_SEP
                    + MemorizeInbox.session_id
                )
                stmt = stmt.where(
                    scope_key.not_in([_SCOPE_SEP.join(k) for k in exclude_sessions])
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

    async def fail_abandoned(self, *, exclude_ids: Collection[int] = ()) -> int:
        """Mark every ``processing`` row outside ``exclude_ids`` as ``failed``.

        The worker passes the rows it is running; any other ``processing``
        row lost its status write and its outcome is unknown.
        """
        stmt = (
            update(MemorizeInbox)
            .where(MemorizeInbox.status == "processing")
            .values(status="failed", last_error=LOST_STATUS_WRITE)
        )
        if exclude_ids:
            stmt = stmt.where(MemorizeInbox.id.not_in(list(exclude_ids)))  # type: ignore[union-attr]
        async with session_scope(self._factory) as s:
            result = await s.execute(stmt)
            await s.commit()
            return int(result.rowcount or 0)

    async def requeue_processing(self, *, max_replays: int = 3) -> tuple[int, int]:
        """Startup recovery: return ``processing`` rows to ``queued``, bounded.

        A row left ``processing`` was in flight when the process died; it is
        delivered again (at-least-once) and ``attempt`` counts that replay.
        A row already replayed ``max_replays`` times is marked ``failed``
        instead, so a batch that kills the process cannot crash-loop it.
        Returns ``(requeued, given_up)``.
        """
        async with session_scope(self._factory) as s:
            gave_up = await s.execute(
                update(MemorizeInbox)
                .where(
                    MemorizeInbox.status == "processing",
                    MemorizeInbox.attempt >= max_replays,
                )
                .values(status="failed", last_error=replays_exhausted(max_replays))
            )
            requeued = await s.execute(
                update(MemorizeInbox)
                .where(MemorizeInbox.status == "processing")
                .values(status="queued", attempt=MemorizeInbox.attempt + 1)
            )
            await s.commit()
            return int(requeued.rowcount or 0), int(gave_up.rowcount or 0)

    async def count_status(self, status: str) -> int:
        async with session_scope(self._factory) as s:
            stmt = (
                select(func.count())
                .select_from(MemorizeInbox)
                .where(MemorizeInbox.status == status)
            )
            return int((await s.execute(stmt)).scalar_one())

    async def oldest_pending_created_at(self) -> dt.datetime | None:
        async with session_scope(self._factory) as s:
            stmt = select(func.min(MemorizeInbox.created_at)).where(
                MemorizeInbox.status.in_(PENDING_STATUSES)  # type: ignore[attr-defined]
            )
            return (await s.execute(stmt)).scalar_one()

    async def list_tasks(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[MemorizeInbox], int]:
        """Rows for ``GET /tasks``: newest first, plus the unpaged total.

        ``session_id`` (with its ``app_id`` / ``project_id`` scope) and
        ``status`` narrow the listing; either may be omitted.
        """
        where = []
        if session_id is not None:
            where += [
                MemorizeInbox.app_id == app_id,
                MemorizeInbox.project_id == project_id,
                MemorizeInbox.session_id == session_id,
            ]
        if status is not None:
            where.append(MemorizeInbox.status == status)
        async with session_scope(self._factory) as s:
            total = (
                await s.execute(
                    select(func.count()).select_from(MemorizeInbox).where(*where)
                )
            ).scalar_one()
            rows = (
                (
                    await s.execute(
                        select(MemorizeInbox)
                        .where(*where)
                        .order_by(MemorizeInbox.id.desc())  # type: ignore[union-attr]
                        .offset(offset)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return list(rows), int(total)

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
