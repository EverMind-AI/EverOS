"""``memory_strength`` repository: read a search's strengths, write a batch.

A search reads the strengths of the memories it ranks, owner by owner (a
search spans one owner, or a user and the user's groups); the reinforcement
writer replaces the rows of one batch in one transaction. Rows are never
deleted.
"""

from __future__ import annotations

__all__ = ["MemoryStrengthRepo", "StrengthRow"]

from collections.abc import Collection, Sequence
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import Insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import MemoryStrength

_SQLITE_IN_CHUNK = 500
"""Memory keys per ``IN`` list, and rows per ``INSERT``: well under SQLite's
bound-parameter limit."""

type Owner = tuple[str, str, str, str]
"""``(app_id, project_id, owner_type, owner_id)``."""


class StrengthRow(NamedTuple):
    """One row as the repository reads and writes it."""

    app_id: str
    project_id: str
    owner_type: str
    owner_id: str
    memory_key: str
    strength_days: float
    reinforced_at_ms: int
    reinforcements: int


class MemoryStrengthRepo(RepoBase[MemoryStrength]):
    """The ``memory_strength`` table."""

    model = MemoryStrength

    async def read(
        self, owner: Owner, memory_keys: Collection[str]
    ) -> list[StrengthRow]:
        """The rows of ``memory_keys`` in one owner's space that exist."""
        app_id, project_id, owner_type, owner_id = owner
        keys = sorted(set(memory_keys))
        found: list[StrengthRow] = []
        async with session_scope(self._factory) as s:
            for start in range(0, len(keys), _SQLITE_IN_CHUNK):
                chunk = keys[start : start + _SQLITE_IN_CHUNK]
                stmt = select(MemoryStrength).where(
                    MemoryStrength.app_id == app_id,  # type: ignore[arg-type]
                    MemoryStrength.project_id == project_id,  # type: ignore[arg-type]
                    MemoryStrength.owner_type == owner_type,  # type: ignore[arg-type]
                    MemoryStrength.owner_id == owner_id,  # type: ignore[arg-type]
                    MemoryStrength.memory_key.in_(chunk),  # type: ignore[attr-defined]
                )
                found.extend(
                    StrengthRow(
                        row.app_id,
                        row.project_id,
                        row.owner_type,
                        row.owner_id,
                        row.memory_key,
                        row.strength_days,
                        row.reinforced_at_ms,
                        row.reinforcements,
                    )
                    for row in (await s.execute(stmt)).scalars()
                )
        return found

    async def write(self, rows: Sequence[StrengthRow]) -> None:
        """Insert or replace ``rows`` in one transaction."""
        if not rows:
            return
        now = get_utc_now()
        async with session_scope(self._factory) as s:
            for start in range(0, len(rows), _SQLITE_IN_CHUNK):
                chunk = rows[start : start + _SQLITE_IN_CHUNK]
                await s.execute(
                    _upsert([r._asdict() | {"updated_at": now} for r in chunk])
                )
            await s.commit()


def _upsert(values: list[dict[str, object]]) -> Insert:
    """``INSERT ... ON CONFLICT DO UPDATE`` of ``values`` (a new row also gets
    ``created_at``; a replaced one keeps it)."""
    stmt = sqlite_insert(MemoryStrength).values(
        [v | {"created_at": v["updated_at"]} for v in values]
    )
    return stmt.on_conflict_do_update(
        index_elements=[
            "app_id",
            "project_id",
            "owner_type",
            "owner_id",
            "memory_key",
        ],
        set_={
            "strength_days": stmt.excluded.strength_days,
            "reinforced_at_ms": stmt.excluded.reinforced_at_ms,
            "reinforcements": stmt.excluded.reinforcements,
            "updated_at": stmt.excluded.updated_at,
        },
    )
