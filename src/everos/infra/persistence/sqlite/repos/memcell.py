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

from everos.infra.persistence.sqlite.base.repository import RepoBase
from everos.infra.persistence.sqlite.base.session import session_scope

from ..tables import Memcell

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
        ``SQLITE_MAX_VARIABLE_NUMBER`` blowups on large owner histories
        (e.g. ``extract_user_profile``'s direct path when ``user.md``
        doesn't exist yet — ``last_profile_ts=0`` fetches every memcell
        the owner has ever produced).

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

    async def mark_case_extracted(
        self,
        memcell_id: str,
        *,
        extracted_at: dt.datetime,
        skip_reason: str | None,
        skip_detail_json: str | None,
    ) -> None:
        """Stamp the agent-case extraction outcome; an unknown id updates nothing."""
        stmt = (
            update(Memcell)
            .where(Memcell.memcell_id == memcell_id)
            .values(
                case_extracted_at=extracted_at,
                case_skip_reason=skip_reason,
                case_skip_detail_json=skip_detail_json,
            )
        )
        async with session_scope(self._factory) as s:
            await s.execute(stmt)
            await s.commit()
