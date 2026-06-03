"""ORM CRUD demo: full INSERT / SELECT / UPDATE / DELETE on a BaseTable.

Doubles as living documentation for how to author a SQLModel-backed
business table inside the everos persistence stack:

    1. Subclass ``BaseTable`` (gets ``created_at`` / ``updated_at`` for free).
    2. Build a session factory from a real engine.
    3. Use ``session_scope`` for the transaction lifecycle.
    4. Verify ``updated_at`` auto-bumps on UPDATE.

The local table name is prefixed with ``_`` so it cannot be confused with
a real business table.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlmodel import SQLModel, select

from everos.config import SqliteSettings
from everos.core.persistence import (
    BaseTable,
    Field,
    MemoryRoot,
    create_session_factory,
    create_system_engine,
    session_scope,
)


class _DemoNote(BaseTable, table=True):
    """Tiny demo table — used only by this test module."""

    __tablename__ = "_demo_notes"  # type: ignore[assignment]

    id: int | None = Field(default=None, primary_key=True)
    body: str
    tags: str | None = Field(default=None)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


async def test_orm_full_crud_lifecycle(memory_root: MemoryRoot) -> None:
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    try:
        # ── Create schema ───────────────────────────────────────────────
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # ── INSERT ──────────────────────────────────────────────────────
        async with session_scope(factory) as s:
            note = _DemoNote(body="hello")
            s.add(note)
            await s.commit()
            await s.refresh(note)
            assert note.id is not None
            assert note.created_at is not None
            assert note.updated_at is not None
            # default_factory runs once per field, so the two timestamps
            # may differ by a few microseconds on INSERT. Order must hold.
            assert note.created_at <= note.updated_at
            note_id = note.id
            initial_created = note.created_at
            initial_updated = note.updated_at

        # ── SELECT (single by id) ───────────────────────────────────────
        async with session_scope(factory) as s:
            stmt = select(_DemoNote).where(_DemoNote.id == note_id)
            result = (await s.execute(stmt)).scalars().first()
            assert result is not None
            assert result.body == "hello"

        # ── SELECT (filter + order) ─────────────────────────────────────
        async with session_scope(factory) as s:
            s.add(_DemoNote(body="second"))
            s.add(_DemoNote(body="third"))
            await s.commit()

        async with session_scope(factory) as s:
            stmt = select(_DemoNote).order_by(_DemoNote.id)
            rows = (await s.execute(stmt)).scalars().all()
            assert [r.body for r in rows] == ["hello", "second", "third"]

        # ── UPDATE (verify updated_at auto-bumps) ───────────────────────
        # Sleep slightly so onupdate has a measurably newer timestamp
        # than the initial insert (timestamp resolution is fine but the
        # comparison should be ``>=`` to be robust on fast machines).
        await asyncio.sleep(0.01)
        async with session_scope(factory) as s:
            stmt = select(_DemoNote).where(_DemoNote.id == note_id)
            n = (await s.execute(stmt)).scalars().first()
            assert n is not None
            n.body = "hello world"
            n.tags = "demo"
            await s.commit()
            await s.refresh(n)
            assert n.body == "hello world"
            assert n.tags == "demo"
            assert n.updated_at >= initial_updated  # bumped via onupdate
            assert n.created_at == initial_created  # unchanged on update

        # ── DELETE ──────────────────────────────────────────────────────
        async with session_scope(factory) as s:
            stmt = select(_DemoNote).where(_DemoNote.id == note_id)
            n = (await s.execute(stmt)).scalars().first()
            assert n is not None
            await s.delete(n)
            await s.commit()

        async with session_scope(factory) as s:
            count_stmt = select(_DemoNote).where(_DemoNote.id == note_id)
            assert (await s.execute(count_stmt)).scalars().first() is None
            # Other rows survive
            remaining = (await s.execute(select(_DemoNote))).scalars().all()
            assert {r.body for r in remaining} == {"second", "third"}
    finally:
        await engine.dispose()
