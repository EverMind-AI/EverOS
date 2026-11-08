"""Repository for ``forget_log`` — append a forget, read it back for its undo.

Two operations, both trivial in isolation and both load-bearing:

- :meth:`ForgetLogRepo.append` writes the record of one forget. It runs **last**
  in a forget, after the marks and the rewrites: a row that exists while the
  forget it describes is half done would make an undo restore material the
  forget never removed.
- :meth:`ForgetLogRepo.latest_live` finds the newest forget of one target that
  no undo has reversed. That is what an undo reverses — never an older one,
  because undoing a forget that a later forget superseded would bring back
  material the user asked to forget again.

Rows are never deleted and never edited except for ``undone_at``: the log is
the audit trail of what was hidden and when it came back.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ForgetLog


class ForgetLogRepo(RepoBase[ForgetLog]):
    model = ForgetLog

    async def append(
        self,
        *,
        target_kind: str,
        target_id: str,
        app_id: str,
        project_id: str,
        owner_id: str,
        owner_type: str = "user",
        reason: str | None = None,
        entry_ids: list[str] | None = None,
        episode_ids: list[list[str]] | None = None,
        removed_profile_items: list[list[object]] | None = None,
    ) -> ForgetLog:
        """Record one forget and return the row (its id is the undo's handle)."""
        row = ForgetLog(
            forget_id=f"fg_{uuid.uuid4().hex[:16]}",
            target_kind=target_kind,
            target_id=target_id,
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
            owner_type=owner_type,
            reason=reason,
            entry_ids_json=json.dumps(entry_ids or []),
            episode_ids_json=json.dumps(episode_ids or []),
            removed_profile_items_json=json.dumps(removed_profile_items or []),
            created_at=get_utc_now(),
        )
        return await self.add(row)

    async def latest_live(
        self,
        *,
        target_kind: str,
        target_id: str,
        app_id: str,
        project_id: str,
        owner_id: str,
    ) -> ForgetLog | None:
        """The newest not-yet-undone forget of this target, or ``None``."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(ForgetLog)
                .where(
                    ForgetLog.target_kind == target_kind,  # type: ignore[arg-type]
                    ForgetLog.target_id == target_id,  # type: ignore[arg-type]
                    ForgetLog.app_id == app_id,  # type: ignore[arg-type]
                    ForgetLog.project_id == project_id,  # type: ignore[arg-type]
                    ForgetLog.owner_id == owner_id,  # type: ignore[arg-type]
                    ForgetLog.undone_at.is_(None),  # type: ignore[union-attr]
                )
                .order_by(ForgetLog.created_at.desc())
                .limit(1)
            )
            return (await s.execute(stmt)).scalars().first()

    async def mark_undone(self, forget_id: str) -> bool:
        """Stamp ``undone_at``; whether this call is the one that stamped it.

        A second undo of the same forget returns ``False`` and restores
        nothing, so a retried request cannot put back material a later forget
        removed.
        """
        async with session_scope(self._factory) as s:
            row = await s.get(ForgetLog, forget_id)
            if row is None or row.undone_at is not None:
                return False
            row.undone_at = get_utc_now()
            await s.commit()
        return True
