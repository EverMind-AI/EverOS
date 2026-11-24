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

from sqlalchemy import bindparam, delete, insert, text, update
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

    async def held_by(
        self,
        session_id: str,
        track: str,
        message_ids: Sequence[str],
        *,
        app_id: str,
        project_id: str,
    ) -> dict[str, str]:
        """The memcell of the (session, track) holding each of ``message_ids``.

        The memcells a cut inserts are its record of what it consumed, in the
        same transaction, so a message has been cut exactly when one of them
        lists it. A message is in one memcell; were it in two, either is
        returned.
        """
        if not message_ids:
            return {}
        # ponytail: scans the session's memcells; a (session, message_id)
        # table if one session ever holds enough memcells for this to show.
        stmt = text(
            "SELECT j.value, m.memcell_id FROM memcell AS m, "
            "json_each(m.message_ids_json) AS j "
            "WHERE m.app_id = :app_id AND m.project_id = :project_id "
            "AND m.session_id = :session_id AND m.track = :track "
            "AND j.value IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        params = _keys(app_id, project_id, session_id, track) | {
            "ids": list(message_ids)
        }
        async with session_scope(self._factory) as s:
            return {row[0]: row[1] for row in await s.execute(stmt, params)}


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
