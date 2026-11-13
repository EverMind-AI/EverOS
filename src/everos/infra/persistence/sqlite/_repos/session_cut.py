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

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ConversationStatus, Memcell, UnprocessedBuffer

_CHUNK = 500
"""Sessions per statement of :meth:`SessionCutRepo.newest_timestamp`."""


class SessionCutRepo(RepoBase[UnprocessedBuffer]):
    """A session's buffer, status stamp and new memcells, written atomically."""

    model = UnprocessedBuffer

    async def newest_timestamp(
        self, app_id: str, project_id: str, session_ids: Sequence[str], track: str
    ) -> dt.datetime | None:
        """The newest message time ``session_ids`` hold on ``track``: a
        buffered message, or the closing message of a memcell cut from them.

        One ``max`` per table and chunk of sessions, each over the
        ``(app, project, session, track, timestamp)`` index.
        """
        wanted = sorted(set(session_ids))
        newest: dt.datetime | None = None
        async with session_scope(self._factory) as s:
            for table in (UnprocessedBuffer, Memcell):
                for start in range(0, len(wanted), _CHUNK):
                    found = await s.scalar(
                        select(func.max(table.timestamp)).where(
                            table.app_id == app_id,  # type: ignore[arg-type]
                            table.project_id == project_id,  # type: ignore[arg-type]
                            table.session_id.in_(  # type: ignore[attr-defined]
                                wanted[start : start + _CHUNK]
                            ),
                            table.track == track,  # type: ignore[arg-type]
                        )
                    )
                    if found is not None and (newest is None or found > newest):
                        newest = found
        return newest

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
        keys = _keys(app_id, project_id, session_id, track)
        async with session_scope(self._factory) as s:
            await _stamp(s, keys, "last_message_ts", last_message_ts)
            await _replace_buffer(s, keys, rows)
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
        keys = _keys(app_id, project_id, session_id, track)
        async with session_scope(self._factory) as s:
            s.add_all(memcells)
            await s.flush()
            if last_memcell_ts is not None:
                await _stamp(s, keys, "last_memcell_ts", last_memcell_ts)
            await _replace_buffer(s, keys, tail)
            await s.commit()


def _keys(app_id: str, project_id: str, session_id: str, track: str) -> dict[str, str]:
    return {
        "app_id": app_id,
        "project_id": project_id,
        "session_id": session_id,
        "track": track,
    }


async def _stamp(
    s: AsyncSession, keys: dict[str, str], column: str, ts: dt.datetime
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
    s: AsyncSession, keys: dict[str, str], rows: Sequence[UnprocessedBuffer]
) -> None:
    await s.execute(
        delete(UnprocessedBuffer).where(
            *(getattr(UnprocessedBuffer, k) == v for k, v in keys.items())
        )
    )
    s.add_all(rows)
    await s.flush()
