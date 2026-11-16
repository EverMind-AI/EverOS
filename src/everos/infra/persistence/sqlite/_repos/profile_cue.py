"""``profile_cue`` repository: a segment's cues written once, read per person.

The conversation pipeline writes one segment's rows in one transaction (a
retried extraction writes them again: the marks are replaced, ``read_at``
kept); a profile update reads one person's rows of the segments it reads and
stamps them read.
"""

from __future__ import annotations

__all__ = ["CueRow", "ProfileCueRepo"]

from collections.abc import Collection, Sequence
from typing import NamedTuple

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ProfileCue

_SQLITE_IN_CHUNK = 500
"""Memcell ids per ``IN`` list: well under SQLite's bound-parameter limit."""


class CueRow(NamedTuple):
    """One row as the repository reads and writes it."""

    memcell_id: str
    person_id: str
    message_ids_json: str
    image_ids_json: str
    read: bool = False


class ProfileCueRepo(RepoBase[ProfileCue]):
    """The ``profile_cue`` table."""

    model = ProfileCue

    async def write(
        self,
        *,
        app_id: str,
        project_id: str,
        session_id: str,
        rows: Sequence[CueRow],
    ) -> None:
        """Insert or replace one segment's ``rows`` in one transaction; a
        replaced row keeps its ``read_at`` and ``created_at``."""
        if not rows:
            return
        now = get_utc_now()
        values = [
            {
                "memcell_id": row.memcell_id,
                "person_id": row.person_id,
                "app_id": app_id,
                "project_id": project_id,
                "session_id": session_id,
                "message_ids_json": row.message_ids_json,
                "image_ids_json": row.image_ids_json,
                "created_at": now,
                "updated_at": now,
            }
            for row in rows
        ]
        stmt = sqlite_insert(ProfileCue).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["memcell_id", "person_id"],
            set_={
                "message_ids_json": stmt.excluded.message_ids_json,
                "image_ids_json": stmt.excluded.image_ids_json,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        async with session_scope(self._factory) as s:
            await s.execute(stmt)
            await s.commit()

    async def of_person(
        self, person_id: str, memcell_ids: Collection[str]
    ) -> list[CueRow]:
        """The person's rows among ``memcell_ids``."""
        ids = list(dict.fromkeys(memcell_ids))
        found: list[CueRow] = []
        async with session_scope(self._factory) as s:
            for start in range(0, len(ids), _SQLITE_IN_CHUNK):
                chunk = ids[start : start + _SQLITE_IN_CHUNK]
                stmt = select(ProfileCue).where(
                    ProfileCue.person_id == person_id,  # type: ignore[arg-type]
                    ProfileCue.memcell_id.in_(chunk),  # type: ignore[attr-defined]
                )
                found.extend(
                    CueRow(
                        row.memcell_id,
                        row.person_id,
                        row.message_ids_json,
                        row.image_ids_json,
                        row.read_at is not None,
                    )
                    for row in (await s.execute(stmt)).scalars()
                )
        return found

    async def mark_read(self, person_id: str, memcell_ids: Collection[str]) -> None:
        """Stamp the person's unread rows among ``memcell_ids`` read."""
        ids = list(dict.fromkeys(memcell_ids))
        if not ids:
            return
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            for start in range(0, len(ids), _SQLITE_IN_CHUNK):
                chunk = ids[start : start + _SQLITE_IN_CHUNK]
                await s.execute(
                    update(ProfileCue)
                    .where(
                        ProfileCue.person_id == person_id,  # type: ignore[arg-type]
                        ProfileCue.memcell_id.in_(chunk),  # type: ignore[attr-defined]
                        ProfileCue.read_at.is_(None),  # type: ignore[union-attr]
                    )
                    .values(read_at=now, updated_at=now)
                )
            await s.commit()
