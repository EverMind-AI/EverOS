"""Repository for ``memorized_message`` (local patch) — see the table module."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from everos.core.persistence.sqlite import RepoBase, session_scope

from ..sqlite_manager import get_session_factory
from ..tables import MemorizedMessage

_SQLITE_IN_CHUNK = 500
"""Keeps ``IN (...)`` under SQLite's default 999-variable cap."""


class _MemorizedMessageRepo(RepoBase[MemorizedMessage]):
    model = MemorizedMessage

    def _factory_lookup(self) -> async_sessionmaker[AsyncSession]:
        return get_session_factory()

    async def find_memorized(
        self,
        session_id: str,
        timestamps_ms: Iterable[int],
        *,
        app_id: str,
        project_id: str,
    ) -> set[int]:
        """Return the subset of ``timestamps_ms`` already in a memcell."""
        wanted = sorted(set(timestamps_ms))
        found: set[int] = set()
        async with session_scope(self._factory) as s:
            for start in range(0, len(wanted), _SQLITE_IN_CHUNK):
                chunk = wanted[start : start + _SQLITE_IN_CHUNK]
                rows = await s.execute(
                    select(MemorizedMessage.timestamp_ms).where(
                        MemorizedMessage.app_id == app_id,
                        MemorizedMessage.project_id == project_id,
                        MemorizedMessage.session_id == session_id,
                        MemorizedMessage.timestamp_ms.in_(chunk),  # type: ignore[attr-defined]
                    )
                )
                found.update(rows.scalars().all())
        return found

    async def record(
        self,
        session_id: str,
        timestamps_ms: Iterable[int],
        *,
        app_id: str,
        project_id: str,
    ) -> None:
        """Insert-or-ignore one row per timestamp."""
        values = [
            {
                "app_id": app_id,
                "project_id": project_id,
                "session_id": session_id,
                "timestamp_ms": ts,
            }
            for ts in sorted(set(timestamps_ms))
        ]
        if not values:
            return
        async with session_scope(self._factory) as s:
            for start in range(0, len(values), _SQLITE_IN_CHUNK // 4):
                await s.execute(
                    insert(MemorizedMessage)
                    .values(values[start : start + _SQLITE_IN_CHUNK // 4])
                    .on_conflict_do_nothing()
                )
            await s.commit()


memorized_message_repo = _MemorizedMessageRepo()
