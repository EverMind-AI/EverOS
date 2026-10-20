"""Repository for ``unprocessed_buffer`` — chat message accumulator.

Singleton bound to the process-wide ``sqlite_manager`` session factory.

Pure SQLModel persistence: row ↔ domain conversion lives in
``everos.memory.extract.pipeline`` (the only caller that needs it).

Exposes:

- :meth:`list_for_track` — load all rows of (session_id, track), ordered by ts.
- :meth:`replace` — atomically swap all rows of (session_id, track) for a
  freshly-built list of :class:`UnprocessedBuffer` rows.
"""

from __future__ import annotations

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import UnprocessedBuffer


class _UnprocessedBufferRepo(RepoBase[UnprocessedBuffer]):
    model = UnprocessedBuffer

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def list_for_track(
        self,
        session_id: str,
        track: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[UnprocessedBuffer]:
        """Return all rows of (app, project, session, track), ts asc."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(UnprocessedBuffer)
                .where(
                    UnprocessedBuffer.app_id == app_id,
                    UnprocessedBuffer.project_id == project_id,
                    UnprocessedBuffer.session_id == session_id,
                    UnprocessedBuffer.track == track,
                )
                .order_by(UnprocessedBuffer.timestamp.asc())  # type: ignore[union-attr]
            )
            return list((await s.execute(stmt)).scalars().all())

    async def replace(
        self,
        session_id: str,
        track: str,
        rows: list[UnprocessedBuffer],
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> None:
        """Atomically rewrite all rows of (app, project, session, track).

        Delete-then-insert in one transaction. Empty ``rows`` clears the slice.
        The delete is scoped to the same (app, project) as the incoming rows so
        one space's buffer never wipes another's.
        """
        async with session_scope(self._factory) as s:
            await s.execute(
                delete(UnprocessedBuffer).where(
                    UnprocessedBuffer.app_id == app_id,
                    UnprocessedBuffer.project_id == project_id,
                    UnprocessedBuffer.session_id == session_id,
                    UnprocessedBuffer.track == track,
                )
            )
            if rows:
                s.add_all(rows)
            await s.commit()

    async def quarantine(
        self,
        session_id: str,
        from_track: str,
        to_track: str,
        extra_rows: list[UnprocessedBuffer],
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> int:
        """Move the (session, from_track) slice to ``to_track`` and add ``extra_rows``.

        One transaction: the rows already buffered change track in place, and
        the rows of the call that triggered the move (not yet persisted) are
        inserted under ``to_track``; an id already present is left as it is.
        Returns the number of rows now set aside by this call.
        """
        async with session_scope(self._factory) as s:
            moved = await s.execute(
                update(UnprocessedBuffer)
                .where(
                    UnprocessedBuffer.app_id == app_id,
                    UnprocessedBuffer.project_id == project_id,
                    UnprocessedBuffer.session_id == session_id,
                    UnprocessedBuffer.track == from_track,
                )
                .values(track=to_track)
            )
            inserted = 0
            for row in extra_rows:
                values = {
                    c.name: getattr(row, c.name)
                    for c in UnprocessedBuffer.__table__.columns  # type: ignore[attr-defined]
                    if getattr(row, c.name) is not None
                }
                values["track"] = to_track
                result = await s.execute(
                    sqlite_insert(UnprocessedBuffer)
                    .values(**values)
                    .on_conflict_do_nothing()
                )
                inserted += result.rowcount
            await s.commit()
        return int(moved.rowcount) + inserted


unprocessed_buffer_repo = _UnprocessedBufferRepo()
