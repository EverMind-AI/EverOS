"""Delete one purge scope's rows from ``system.db``."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.sql import ColumnElement

from everos.core.purge_marks import PURGE_DISCARD_PREFIX, PurgeScope
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import (
    Cluster,
    ClusterMember,
    ConversationStatus,
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
    MdChangeState,
    Memcell,
    MemorizeInbox,
    ReflectionReport,
    UnprocessedBuffer,
)

_SCOPED: tuple[Any, ...] = (
    Memcell,
    UnprocessedBuffer,
    ConversationStatus,
    ReflectionReport,
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
)
"""Tables with ``app_id`` / ``project_id``; ``cluster`` goes with its members."""

INBOX_DISCARD_PREFIX = PURGE_DISCARD_PREFIX


def _in_scope(table: Any, scope: PurgeScope) -> list[ColumnElement[bool]]:
    if scope.kind == "space":
        return []
    where = [table.app_id == scope.app_id]
    if scope.kind == "project":
        where.append(table.project_id == scope.project_id)
    return where


def _change_state_prefix(scope: PurgeScope) -> str | None:
    """``<app_dir>/`` or ``<app_dir>/<project_dir>/`` (``None``: the root).

    The trailing ``/`` keeps ``a1/p1`` from matching ``a1/p10``; a plain
    prefix compare (not ``LIKE``) keeps ``_`` in a name literal.
    """
    if scope.kind == "space":
        return None
    if scope.kind == "app":
        return f"{scope.app_dir}/"
    return f"{scope.app_dir}/{scope.project_dir}/"


class ScopePurgeRepo(RepoBase[Memcell]):
    """Count and delete a purge scope's rows (``purge_job`` itself excepted)."""

    model = Memcell

    async def count(self, scope: PurgeScope) -> dict[str, int]:
        """Rows per table, ``cluster_member`` included."""
        counts: dict[str, int] = {}
        async with session_scope(self._factory) as s:
            for table in (*_SCOPED, Cluster):
                counts[table.__tablename__] = int(
                    await s.scalar(
                        select(func.count())
                        .select_from(table)
                        .where(*_in_scope(table, scope))
                    )
                    or 0
                )
            counts[ClusterMember.__tablename__] = int(
                await s.scalar(
                    select(func.count())
                    .select_from(ClusterMember)
                    .where(ClusterMember.cluster_id.in_(self._clusters(scope)))
                )
                or 0
            )
        return counts

    async def delete(self, scope: PurgeScope) -> dict[str, int]:
        """Delete the scope's rows in one transaction; rows deleted per table."""
        deleted: dict[str, int] = {}
        async with session_scope(self._factory) as s:
            result = await s.execute(
                delete(ClusterMember).where(
                    ClusterMember.cluster_id.in_(self._clusters(scope))
                )
            )
            deleted[ClusterMember.__tablename__] = result.rowcount
            for table in (Cluster, *_SCOPED):
                result = await s.execute(delete(table).where(*_in_scope(table, scope)))
                deleted[table.__tablename__] = result.rowcount
            await s.commit()
        return deleted

    async def discard_queued_inbox(
        self, scope: PurgeScope, *, job_id: str, before: datetime
    ) -> int:
        """Fail the scope's async requests queued before ``before``, bodies dropped."""
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(MemorizeInbox)
                .where(
                    MemorizeInbox.status == "queued",
                    MemorizeInbox.created_at < before,
                    *_in_scope(MemorizeInbox, scope),
                )
                .values(
                    status="failed",
                    payload_json="",
                    last_error=(
                        f"{INBOX_DISCARD_PREFIX}discarded by purge {job_id};"
                        " do not resend"
                    ),
                )
            )
            await s.commit()
            return result.rowcount

    async def delete_inbox_history(self, scope: PurgeScope, *, before: datetime) -> int:
        """Delete the scope's requests finished from before ``before``, except
        the ones a purge discarded (``/tasks`` must still say why)."""
        prefix = INBOX_DISCARD_PREFIX
        async with session_scope(self._factory) as s:
            result = await s.execute(
                delete(MemorizeInbox).where(
                    MemorizeInbox.status.in_(("success", "failed")),
                    MemorizeInbox.created_at < before,
                    func.coalesce(
                        func.substr(MemorizeInbox.last_error, 1, len(prefix)), ""
                    )
                    != prefix,
                    *_in_scope(MemorizeInbox, scope),
                )
            )
            await s.commit()
            return result.rowcount

    async def delete_buffer(self, scope: PurgeScope) -> int:
        """Delete the scope's unprocessed messages (before anything flushes them)."""
        async with session_scope(self._factory) as s:
            result = await s.execute(
                delete(UnprocessedBuffer).where(*_in_scope(UnprocessedBuffer, scope))
            )
            await s.commit()
            return result.rowcount

    async def delete_change_state(self, scope: PurgeScope) -> int:
        """Forget the scope's md change records (the scanner re-registers what
        is still on disk)."""
        prefix = _change_state_prefix(scope)
        statement = delete(MdChangeState)
        if prefix is not None:
            statement = statement.where(
                func.substr(MdChangeState.md_path, 1, len(prefix)) == prefix
            )
        async with session_scope(self._factory) as s:
            result = await s.execute(statement)
            await s.commit()
            return result.rowcount

    @staticmethod
    def _clusters(scope: PurgeScope) -> Any:
        return select(Cluster.cluster_id).where(*_in_scope(Cluster, scope))
