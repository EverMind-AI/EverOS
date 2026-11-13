"""Repository for ``memcell`` table.

Pure persistence: callers build the SQLModel ``Memcell`` rows (including
``message_ids_json`` / ``sender_ids_json``) and hand them in. The pipeline
is responsible for mapping algo-side messages back to everos
``message_id`` because algo's ``Message`` does not carry per-message
identifiers.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select, update

from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import Memcell

_SQLITE_IN_CHUNK = 500
"""SQL ``IN (?, ?, ...)`` parameter cap for :meth:`MemcellRepo.find_by_ids`.

SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` defaults to 999 on the shipped
amalgamation and rises to 32766 on ≥3.32; 500 stays well under the older
cap and keeps the query planner's cost estimate cheap. Callers passing
more ``memcell_ids`` than this fan out into multiple SELECTs, merged
in-memory — see the direct-path selector in
``memory/strategies/extract_user_profile.py`` where
``last_profile_ts=0`` can pull an owner's entire episode history."""


class MemcellRepo(RepoBase[Memcell]):
    model = Memcell

    async def insert_many(self, rows: list[Memcell]) -> list[Memcell]:
        """Insert MemCell rows in one transaction; rows are constructed by caller."""
        async with session_scope(self._factory) as s:
            s.add_all(rows)
            await s.commit()
            for r in rows:
                await s.refresh(r)
        return rows

    async def find_by_ids(self, memcell_ids: list[str]) -> list[Memcell]:
        """Bulk fetch rows by primary key list — preserves caller order.

        Chunked at ``_SQLITE_IN_CHUNK`` to protect against
        ``SQLITE_MAX_VARIABLE_NUMBER`` blowups on large id lists (a whole
        cluster's members, a forgotten conversation's segments).

        Used by offline strategies that pull every memcell in a cluster
        (membership lives in :class:`ClusterMember` and is supplied to
        the strategy via :class:`everalgo.clustering.Cluster.members`).
        """
        if not memcell_ids:
            return []
        by_id: dict[str, Memcell] = {}
        async with session_scope(self._factory) as s:
            for start in range(0, len(memcell_ids), _SQLITE_IN_CHUNK):
                chunk = memcell_ids[start : start + _SQLITE_IN_CHUNK]
                stmt = select(Memcell).where(Memcell.memcell_id.in_(chunk))
                rows = (await s.execute(stmt)).scalars().all()
                by_id.update((r.memcell_id, r) for r in rows)
        return [by_id[mid] for mid in memcell_ids if mid in by_id]

    async def set_retry(self, memcell_id: str, retry_json: str | None) -> None:
        """Write one row's ``extraction_retry`` (``None`` clears it).

        A row that is gone (its session deleted) is left gone.
        """
        stmt = (
            update(Memcell)
            .where(Memcell.memcell_id == memcell_id)  # type: ignore[arg-type]
            .values(extraction_retry=retry_json)
        )
        async with session_scope(self._factory) as s:
            await s.execute(stmt)
            await s.commit()

    async def retries_in_session(
        self, *, app_id: str, project_id: str, session_id: str
    ) -> list[Memcell]:
        """The session's rows whose ``extraction_retry`` is set, oldest first."""
        stmt = (
            select(Memcell)
            .where(
                Memcell.app_id == app_id,
                Memcell.project_id == project_id,
                Memcell.session_id == session_id,
                Memcell.extraction_retry.is_not(None),  # type: ignore[union-attr]
            )
            .order_by(
                Memcell.timestamp,  # type: ignore[arg-type]
                Memcell.memcell_id,  # type: ignore[arg-type]
            )
        )
        async with session_scope(self._factory) as s:
            return list((await s.execute(stmt)).scalars().all())

    async def senders_in_session(
        self, *, app_id: str, project_id: str, session_id: str
    ) -> list[str]:
        """Every sender of the session's rows, as their distinct JSON lists.

        One ``DISTINCT`` over the ``(app, project, session, ...)`` index: a
        conversation's rows repeat a handful of sender lists, so a long one
        costs no more than a short one to read.
        """
        stmt = (
            select(Memcell.sender_ids_json)
            .where(
                Memcell.app_id == app_id,
                Memcell.project_id == project_id,
                Memcell.session_id == session_id,
            )
            .distinct()
        )
        async with session_scope(self._factory) as s:
            return list((await s.execute(stmt)).scalars().all())

    async def in_session(
        self, *, app_id: str, project_id: str, session_id: str
    ) -> list[Memcell]:
        """Every row of the session, oldest first (ties by ``memcell_id``)."""
        stmt = (
            select(Memcell)
            .where(
                Memcell.app_id == app_id,
                Memcell.project_id == project_id,
                Memcell.session_id == session_id,
            )
            .order_by(
                Memcell.timestamp,  # type: ignore[arg-type]
                Memcell.memcell_id,  # type: ignore[arg-type]
            )
        )
        async with session_scope(self._factory) as s:
            return list((await s.execute(stmt)).scalars().all())

    async def recent_in_session(
        self,
        *,
        app_id: str,
        project_id: str,
        session_id: str,
        until: dt.datetime,
        limit: int,
        exclude: str | None = None,
    ) -> list[Memcell]:
        """The session's rows up to ``until`` (inclusive), newest first.

        Rides ``ix_memcell_session`` for the scope + session prefix. Every
        track is read: rows written before the ``memorize`` track name carry
        an older one and are the same conversation. Ties on ``timestamp``
        order by ``memcell_id`` so the answer is stable.
        """
        stmt = select(Memcell).where(
            Memcell.app_id == app_id,
            Memcell.project_id == project_id,
            Memcell.session_id == session_id,
            Memcell.timestamp <= until,  # type: ignore[operator]
        )
        if exclude is not None:
            stmt = stmt.where(Memcell.memcell_id != exclude)
        stmt = stmt.order_by(
            Memcell.timestamp.desc(),  # type: ignore[union-attr]
            Memcell.memcell_id.desc(),  # type: ignore[attr-defined]
        ).limit(limit)
        async with session_scope(self._factory) as s:
            return list((await s.execute(stmt)).scalars().all())
