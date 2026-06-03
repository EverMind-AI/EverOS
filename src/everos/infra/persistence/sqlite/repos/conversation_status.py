"""Repository for ``conversation_status`` — singleton bound to ``sqlite_manager``.

Upsert helpers for the (session_id, track) window pointer.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import ConversationStatus


class _ConversationStatusRepo(RepoBase[ConversationStatus]):
    model = ConversationStatus

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def touch_last_message_ts(
        self,
        session_id: str,
        track: str,
        ts: dt.datetime,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> None:
        """Upsert (app, project, session, track); set ``last_message_ts``."""
        await self._upsert(
            session_id, track, app_id=app_id, project_id=project_id, last_message_ts=ts
        )

    async def touch_last_memcell_ts(
        self,
        session_id: str,
        track: str,
        ts: dt.datetime,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> None:
        """Upsert (app, project, session, track); set ``last_memcell_ts``."""
        await self._upsert(
            session_id, track, app_id=app_id, project_id=project_id, last_memcell_ts=ts
        )

    async def _upsert(
        self,
        session_id: str,
        track: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        last_message_ts: dt.datetime | None = None,
        last_memcell_ts: dt.datetime | None = None,
    ) -> None:
        async with session_scope(self._factory) as s:
            stmt = select(ConversationStatus).where(
                ConversationStatus.app_id == app_id,
                ConversationStatus.project_id == project_id,
                ConversationStatus.session_id == session_id,
                ConversationStatus.track == track,
            )
            existing = (await s.execute(stmt)).scalars().first()
            if existing is None:
                s.add(
                    ConversationStatus(
                        app_id=app_id,
                        project_id=project_id,
                        session_id=session_id,
                        track=track,
                        last_message_ts=last_message_ts,
                        last_memcell_ts=last_memcell_ts,
                    )
                )
            else:
                if last_message_ts is not None:
                    existing.last_message_ts = last_message_ts
                if last_memcell_ts is not None:
                    existing.last_memcell_ts = last_memcell_ts
            await s.commit()


conversation_status_repo = _ConversationStatusRepo()
