"""Repository for the ``sender_name`` table."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import SenderName


class SenderNameRepo(RepoBase[SenderName]):
    """Latest display name per sender, keyed by ``(app_id, project_id, sender_id)``.

    The id lists come from one add request's senders, so a single ``IN``
    stays far below SQLite's parameter cap.
    """

    model = SenderName

    async def get_many(
        self, *, app_id: str, project_id: str, sender_ids: Collection[str]
    ) -> dict[str, str]:
        """``sender_id -> sender_name`` for the ids that have a row."""
        if not sender_ids:
            return {}
        async with session_scope(self._factory) as s:
            rows = await s.execute(
                select(SenderName.sender_id, SenderName.sender_name)
                .where(SenderName.app_id == app_id)
                .where(SenderName.project_id == project_id)
                .where(SenderName.sender_id.in_(sender_ids))
            )
            return {sender_id: name for sender_id, name in rows.all()}

    async def upsert_many(
        self, *, app_id: str, project_id: str, names: Mapping[str, str]
    ) -> None:
        """Insert each name, or overwrite the sender's existing row."""
        if not names:
            return
        now = get_utc_now()
        statement = sqlite_insert(SenderName).values(
            [
                {
                    "app_id": app_id,
                    "project_id": project_id,
                    "sender_id": sender_id,
                    "sender_name": name,
                    "created_at": now,
                    "updated_at": now,
                }
                for sender_id, name in names.items()
            ]
        )
        statement = statement.on_conflict_do_update(
            index_elements=["app_id", "project_id", "sender_id"],
            set_={
                "sender_name": statement.excluded.sender_name,
                "updated_at": now,
            },
        )
        async with session_scope(self._factory) as s:
            await s.execute(statement)
            await s.commit()

    async def delete_one(self, *, app_id: str, project_id: str, sender_id: str) -> None:
        """Remove the sender's row; a missing row is a no-op."""
        async with session_scope(self._factory) as s:
            await s.execute(
                delete(SenderName)
                .where(SenderName.app_id == app_id)
                .where(SenderName.project_id == project_id)
                .where(SenderName.sender_id == sender_id)
            )
            await s.commit()
