"""Delete one session's conversation state, in one transaction."""

from __future__ import annotations

from sqlalchemy import delete, select, update

from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ConversationStatus, Memcell, MemorizeInbox, UnprocessedBuffer


class SessionPurgeRepo(RepoBase[Memcell]):
    """Everything a session left in ``system.db`` before extraction.

    Deletes its memcells, its unprocessed buffer (every track, the
    quarantine one included) and its conversation status, and blanks the
    request body of its failed ``memorize_inbox`` rows. Queued, processing
    and succeeded inbox rows are left as they are: a deletion only runs once
    none of the first two exist, and a succeeded row's body is already gone.
    """

    model = Memcell

    async def purge(
        self, *, app_id: str, project_id: str, session_id: str
    ) -> list[Memcell]:
        """Delete the session's rows; return its memcells as they were."""
        async with session_scope(self._factory) as s:
            memcells = list(
                (
                    await s.execute(
                        select(Memcell)
                        .where(Memcell.app_id == app_id)
                        .where(Memcell.project_id == project_id)
                        .where(Memcell.session_id == session_id)
                    )
                )
                .scalars()
                .all()
            )
            for table in (Memcell, UnprocessedBuffer, ConversationStatus):
                await s.execute(
                    delete(table)
                    .where(table.app_id == app_id)
                    .where(table.project_id == project_id)
                    .where(table.session_id == session_id)
                )
            await s.execute(
                update(MemorizeInbox)
                .where(MemorizeInbox.app_id == app_id)
                .where(MemorizeInbox.project_id == project_id)
                .where(MemorizeInbox.session_id == session_id)
                .where(MemorizeInbox.status == "failed")
                .values(payload_json="")
            )
            await s.commit()
        return memcells
