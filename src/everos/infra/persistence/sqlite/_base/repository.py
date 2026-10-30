"""Generic CRUD repository for SQLModel-backed tables.

``RepoBase`` is a pure generic CRUD helper that sits alongside
:class:`BaseTable`. It knows nothing about a storage runtime: every repo is
built over the session factory it should use.

Each method opens its own ``session_scope`` (auto rollback on exception,
session closed at end). For multi-step transactional work, use the
session factory directly via :attr:`session_factory`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel, select

from .session import session_scope


class RepoBase[T: SQLModel]:
    """Generic CRUD repository for one SQLModel table.

    Subclass and bind to a model; build it over a session factory::

           repo = SenderRepo(session_factory)
           await repo.add(Sender(name="alice"))

    Args:
        session_factory: Opens the sessions every method runs in.
    """

    model: type[T]

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = session_factory

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Underlying session factory (for multi-step transactions)."""
        return self._factory

    # ── Create ─────────────────────────────────────────────────────────────

    async def add(self, instance: T) -> T:
        """Insert one row, commit, refresh, return the instance."""
        async with session_scope(self._factory) as s:
            s.add(instance)
            await s.commit()
            await s.refresh(instance)
        return instance

    async def add_many(self, instances: Sequence[T]) -> list[T]:
        """Insert many rows in one transaction."""
        items = list(instances)
        async with session_scope(self._factory) as s:
            s.add_all(items)
            await s.commit()
            for inst in items:
                await s.refresh(inst)
        return items

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_by_id(self, id_value: Any) -> T | None:
        """Get a row by primary key. Returns ``None`` if not found."""
        async with session_scope(self._factory) as s:
            return await s.get(self.model, id_value)

    async def list_all(self) -> list[T]:
        """Return all rows (no filter, no order)."""
        async with session_scope(self._factory) as s:
            stmt = select(self.model)
            return list((await s.execute(stmt)).scalars().all())

    async def find_where(self, **filters: Any) -> list[T]:
        """Equality-only filtering, e.g. ``find_where(name="alice", active=True)``."""
        async with session_scope(self._factory) as s:
            stmt = select(self.model).filter_by(**filters)
            return list((await s.execute(stmt)).scalars().all())

    async def find_one(self, **filters: Any) -> T | None:
        """First row matching ``filters`` (no ordering); ``None`` if not found."""
        async with session_scope(self._factory) as s:
            stmt = select(self.model).filter_by(**filters).limit(1)
            return (await s.execute(stmt)).scalars().first()

    async def count(self) -> int:
        """Total row count (no filter)."""
        async with session_scope(self._factory) as s:
            stmt = select(func.count()).select_from(self.model)
            return int((await s.execute(stmt)).scalar_one())

    # ── Update ─────────────────────────────────────────────────────────────

    async def update(self, instance: T) -> T:
        """Persist changes on an instance whose primary key already exists.

        Uses ``session.merge`` so detached / fresh-from-Pydantic instances
        are reattached. ``BaseTable.updated_at`` auto-bumps via SA's
        ``onupdate`` hook.
        """
        async with session_scope(self._factory) as s:
            merged = await s.merge(instance)
            await s.commit()
            await s.refresh(merged)
        return merged

    # ── Delete ─────────────────────────────────────────────────────────────

    async def delete(self, instance: T) -> None:
        """Delete by instance (primary key must be set)."""
        async with session_scope(self._factory) as s:
            merged = await s.merge(instance)
            await s.delete(merged)
            await s.commit()

    async def delete_by_id(self, id_value: Any) -> bool:
        """Delete by primary key. Returns ``True`` if a row was removed."""
        async with session_scope(self._factory) as s:
            instance = await s.get(self.model, id_value)
            if instance is None:
                return False
            await s.delete(instance)
            await s.commit()
            return True
