"""Repository of ``group_message``: which messages each group already holds.

A batch asks once for all its keys (:meth:`GroupMessageRepo.held`, one ``IN``
over the primary key) and records the keys it stored in one statement
(:meth:`GroupMessageRepo.record`). A key recorded again -- a message taken
again after a ``/memory/delete`` released its session -- moves to the session
that took it.
"""

from __future__ import annotations

from collections.abc import Collection

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import GroupMessage

_CHUNK = 500
"""Keys per statement; a batch holds at most 500 messages."""


class GroupMessageRepo(RepoBase[GroupMessage]):
    """The ``group_message`` table of one ``system.db``."""

    model = GroupMessage

    async def held(
        self, app_id: str, project_id: str, group_id: str, keys: Collection[str]
    ) -> dict[str, str]:
        """Of ``keys``, those the group holds, each with the session holding it."""
        wanted = sorted(set(keys))
        found: dict[str, str] = {}
        async with session_scope(self._factory) as s:
            for start in range(0, len(wanted), _CHUNK):
                rows = await s.execute(
                    select(GroupMessage.key, GroupMessage.session_id).where(  # type: ignore[call-overload]
                        GroupMessage.app_id == app_id,  # type: ignore[arg-type]
                        GroupMessage.project_id == project_id,  # type: ignore[arg-type]
                        GroupMessage.group_id == group_id,  # type: ignore[arg-type]
                        GroupMessage.key.in_(wanted[start : start + _CHUNK]),  # type: ignore[attr-defined]
                    )
                )
                found.update({str(key): str(session) for key, session in rows.all()})
        return found

    async def record(
        self,
        app_id: str,
        project_id: str,
        group_id: str,
        session_id: str,
        keys: Collection[str],
    ) -> None:
        """Note that ``session_id`` holds ``keys``; a key held before moves to it."""
        wanted = sorted(set(keys))
        if not wanted:
            return
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            for start in range(0, len(wanted), _CHUNK):
                statement = insert(GroupMessage).values(
                    [
                        {
                            "app_id": app_id,
                            "project_id": project_id,
                            "group_id": group_id,
                            "key": key,
                            "session_id": session_id,
                            "created_at": now,
                            "updated_at": now,
                        }
                        for key in wanted[start : start + _CHUNK]
                    ]
                )
                await s.execute(
                    statement.on_conflict_do_update(
                        index_elements=["app_id", "project_id", "group_id", "key"],
                        set_={
                            "session_id": statement.excluded.session_id,
                            "updated_at": statement.excluded.updated_at,
                        },
                    )
                )
            await s.commit()
