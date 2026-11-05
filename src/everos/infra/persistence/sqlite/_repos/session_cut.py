"""Rewrite one session's buffer together with its status, in one transaction.

Boundary detection ends every call by writing up to three tables: the
memcells it cut, the session's ``conversation_status`` stamp, and the
messages left buffered. Written as separate commits, a failure after the
first (a write lock still held past ``busy_timeout``) left memcells that
never reached extraction, or a buffer whose status was never stamped.
Each method here commits all of it or none of it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from sqlalchemy import delete, insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ConversationStatus, Memcell, UnprocessedBuffer


class SessionCutRepo(RepoBase[UnprocessedBuffer]):
    """A session's buffer, status stamp and new memcells, written atomically."""

    model = UnprocessedBuffer

    async def keep(
        self,
        session_id: str,
        track: str,
        rows: Sequence[UnprocessedBuffer],
        *,
        app_id: str,
        project_id: str,
        last_message_ts: dt.datetime,
    ) -> None:
        """Buffer ``rows`` as the whole (session, track) slice; stamp
        ``last_message_ts``."""
        async with session_scope(self._factory) as s:
            await _stamp(
                s,
                session_id,
                track,
                app_id,
                project_id,
                "last_message_ts",
                last_message_ts,
            )
            await _replace_buffer(s, session_id, track, rows, app_id, project_id)
            await s.commit()

    async def cut(
        self,
        session_id: str,
        track: str,
        memcells: Sequence[Memcell],
        tail: Sequence[UnprocessedBuffer],
        *,
        app_id: str,
        project_id: str,
        last_memcell_ts: dt.datetime | None,
    ) -> None:
        """Insert ``memcells``, stamp ``last_memcell_ts`` (when given) and
        leave only ``tail`` buffered."""
        async with session_scope(self._factory) as s:
            s.add_all(memcells)
            await s.flush()
            if last_memcell_ts is not None:
                await _stamp(
                    s,
                    session_id,
                    track,
                    app_id,
                    project_id,
                    "last_memcell_ts",
                    last_memcell_ts,
                )
            await _replace_buffer(s, session_id, track, tail, app_id, project_id)
            await s.commit()


async def _stamp(
    s: AsyncSession,
    session_id: str,
    track: str,
    app_id: str,
    project_id: str,
    column: str,
    ts: dt.datetime,
) -> None:
    """Set ``column`` on the session's status row, creating the row if
    there is none; the other stamp keeps its value.

    An UPDATE then, only when it matched nothing, an INSERT, rather than
    ``INSERT ... ON CONFLICT``: databases created before the
    ``(app_id, project_id, session_id, track)`` unique constraint existed
    still carry an older one, which ``ON CONFLICT`` over these four columns
    would not match. Both run inside the caller's transaction.
    """
    now = get_utc_now()
    keys = {
        "app_id": app_id,
        "project_id": project_id,
        "session_id": session_id,
        "track": track,
    }
    updated = await s.execute(
        update(ConversationStatus)
        .where(*(getattr(ConversationStatus, k) == v for k, v in keys.items()))
        .values({column: ts, "updated_at": now})
    )
    if updated.rowcount == 0:
        await s.execute(
            insert(ConversationStatus).values(
                **keys, created_at=now, updated_at=now, **{column: ts}
            )
        )


async def _replace_buffer(
    s: AsyncSession,
    session_id: str,
    track: str,
    rows: Sequence[UnprocessedBuffer],
    app_id: str,
    project_id: str,
) -> None:
    await s.execute(
        delete(UnprocessedBuffer).where(
            UnprocessedBuffer.app_id == app_id,
            UnprocessedBuffer.project_id == project_id,
            UnprocessedBuffer.session_id == session_id,
            UnprocessedBuffer.track == track,
        )
    )
    s.add_all(rows)
    await s.flush()
