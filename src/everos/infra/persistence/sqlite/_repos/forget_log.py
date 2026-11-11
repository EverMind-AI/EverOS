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
from collections.abc import Collection

from sqlalchemy import select, update

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ForgetLog, Memcell, MessageForget

_SQLITE_IN_CHUNK = 500
"""``IN (...)`` parameter cap, as the memcell repository chunks it."""


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
        requested_by: str | None = None,
        undo_json: str | None = None,
        forget_id: str | None = None,
    ) -> ForgetLog:
        """Record one forget and return the row (its id is the undo's handle)."""
        row = ForgetLog(
            forget_id=forget_id or mint_forget_id(),
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
            requested_by=requested_by,
            undo_json=undo_json,
            created_at=get_utc_now(),
        )
        return await self.add(row)

    async def get(self, forget_id: str) -> ForgetLog | None:
        """One forget by id, undone or not."""
        async with session_scope(self._factory) as s:
            return await s.get(ForgetLog, forget_id)

    async def live_targets(
        self, *, target_kind: str, app_id: str, project_id: str, owner_id: str
    ) -> list[str]:
        """Target ids of the owner's live forgets of one kind, oldest first."""
        async with session_scope(self._factory) as s:
            stmt = (
                select(ForgetLog.target_id)
                .where(
                    ForgetLog.target_kind == target_kind,  # type: ignore[arg-type]
                    ForgetLog.app_id == app_id,  # type: ignore[arg-type]
                    ForgetLog.project_id == project_id,  # type: ignore[arg-type]
                    ForgetLog.owner_id == owner_id,  # type: ignore[arg-type]
                    ForgetLog.undone_at.is_(None),  # type: ignore[union-attr]
                )
                .order_by(ForgetLog.created_at)  # type: ignore[arg-type]
            )
            return list(dict.fromkeys((await s.execute(stmt)).scalars().all()))

    async def holds_picture(self, image_id: str) -> bool:
        """Whether a live group forget's undo record took out a memory of the hash.

        The undo record names each picture memory it took out with an
        ``"image_id": "<hash>"`` pair; a 64-hex hash matches nothing else.
        """
        pattern = f'%"image_id": "{image_id}"%'
        async with session_scope(self._factory) as s:
            stmt = (
                select(ForgetLog.forget_id)
                .where(
                    ForgetLog.owner_type == "group",  # type: ignore[arg-type]
                    ForgetLog.undone_at.is_(None),  # type: ignore[union-attr]
                    ForgetLog.undo_json.like(pattern),  # type: ignore[union-attr]
                )
                .limit(1)
            )
            return (await s.execute(stmt)).first() is not None

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


def mint_forget_id() -> str:
    """A new forget's id (``fg_<16 hex>``)."""
    return f"fg_{uuid.uuid4().hex[:16]}"


class MessageForgetRepo(RepoBase[MessageForget]):
    """``message_forget``: which messages of a session a live forget names."""

    model = MessageForget

    async def record(
        self,
        *,
        app_id: str,
        project_id: str,
        session_id: str,
        message_ids: Collection[str],
        forget_id: str,
    ) -> None:
        """One row per message, in one transaction; a repeat adds nothing."""
        wanted = list(dict.fromkeys(message_ids))
        if not wanted:
            return
        async with session_scope(self._factory) as s:
            for message_id in wanted:
                key = (app_id, project_id, session_id, message_id, forget_id)
                if await s.get(MessageForget, key) is None:
                    s.add(
                        MessageForget(
                            app_id=app_id,
                            project_id=project_id,
                            session_id=session_id,
                            message_id=message_id,
                            forget_id=forget_id,
                        )
                    )
            await s.commit()

    async def undo(self, forget_id: str) -> int:
        """Stamp the forget's live rows undone; how many."""
        stmt = (
            update(MessageForget)
            .where(
                MessageForget.forget_id == forget_id,  # type: ignore[arg-type]
                MessageForget.undone_at.is_(None),  # type: ignore[union-attr]
            )
            .values(undone_at=get_utc_now())
        )
        async with session_scope(self._factory) as s:
            result = await s.execute(stmt)
            await s.commit()
        return int(getattr(result, "rowcount", 0) or 0)

    async def live_in_sessions(
        self, app_id: str, project_id: str, session_ids: Collection[str]
    ) -> dict[str, set[str]]:
        """``session_id -> forgotten message ids`` for the sessions given."""
        found: dict[str, set[str]] = {}
        sessions = list(dict.fromkeys(session_ids))
        async with session_scope(self._factory) as s:
            for start in range(0, len(sessions), _SQLITE_IN_CHUNK):
                chunk = sessions[start : start + _SQLITE_IN_CHUNK]
                stmt = select(MessageForget.session_id, MessageForget.message_id).where(
                    MessageForget.app_id == app_id,  # type: ignore[arg-type]
                    MessageForget.project_id == project_id,  # type: ignore[arg-type]
                    MessageForget.session_id.in_(chunk),  # type: ignore[attr-defined]
                    MessageForget.undone_at.is_(None),  # type: ignore[union-attr]
                )
                for session_id, message_id in (await s.execute(stmt)).all():
                    found.setdefault(session_id, set()).add(message_id)
        return found

    async def sessions_of(
        self, memcell_ids: Collection[str]
    ) -> dict[str, tuple[str, str, str]]:
        """``memcell_id -> (app_id, project_id, session_id)`` of existing cells."""
        ids = list(dict.fromkeys(memcell_ids))
        found: dict[str, tuple[str, str, str]] = {}
        async with session_scope(self._factory) as s:
            for start in range(0, len(ids), _SQLITE_IN_CHUNK):
                chunk = ids[start : start + _SQLITE_IN_CHUNK]
                stmt = select(
                    Memcell.memcell_id,
                    Memcell.app_id,
                    Memcell.project_id,
                    Memcell.session_id,
                ).where(Memcell.memcell_id.in_(chunk))  # type: ignore[attr-defined]
                for memcell_id, app_id, project_id, session_id in (
                    await s.execute(stmt)
                ).all():
                    found[memcell_id] = (app_id, project_id, session_id)
        return found
