"""RepoBase CRUD demo + assertions.

Doubles as living documentation for how a service / memory layer caller
uses the generic repository — no manual session handling. Exercises the
explicit-factory constructor path; the lazy ``_factory_lookup`` hook is
exercised indirectly via the lifespan + manager tests once business
repos land under ``infra/.../repos/``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    BaseTable,
    Field,
    MemoryRoot,
    RepoBase,
    create_session_factory,
    create_system_engine,
)


class _DemoUser(BaseTable, table=True):
    """Demo table — only used by this test module."""

    __tablename__ = "_demo_users"  # type: ignore[assignment]

    id: int | None = Field(default=None, primary_key=True)
    name: str
    active: bool = Field(default=True)


class _DemoUserRepo(RepoBase[_DemoUser]):
    model = _DemoUser


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


async def _setup_repo(memory_root: MemoryRoot) -> tuple[_DemoUserRepo, object]:
    """Build engine, factory, and ensure schema. Returns (repo, engine)."""
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _DemoUserRepo(factory), engine


async def test_repo_add_and_get(memory_root: MemoryRoot) -> None:
    repo, engine = await _setup_repo(memory_root)
    try:
        added = await repo.add(_DemoUser(name="alice"))
        assert added.id is not None
        assert added.created_at is not None

        fetched = await repo.get_by_id(added.id)
        assert fetched is not None
        assert fetched.name == "alice"
        assert fetched.active is True
    finally:
        await engine.dispose()


async def test_repo_add_many_and_list_all(memory_root: MemoryRoot) -> None:
    repo, engine = await _setup_repo(memory_root)
    try:
        users = await repo.add_many(
            [
                _DemoUser(name="alice"),
                _DemoUser(name="bob"),
                _DemoUser(name="carol", active=False),
            ]
        )
        assert all(u.id is not None for u in users)

        all_users = await repo.list_all()
        assert {u.name for u in all_users} == {"alice", "bob", "carol"}

        assert await repo.count() == 3
    finally:
        await engine.dispose()


async def test_repo_find_where_and_find_one(memory_root: MemoryRoot) -> None:
    repo, engine = await _setup_repo(memory_root)
    try:
        await repo.add_many(
            [
                _DemoUser(name="alice", active=True),
                _DemoUser(name="bob", active=False),
                _DemoUser(name="carol", active=True),
            ]
        )

        actives = await repo.find_where(active=True)
        assert {u.name for u in actives} == {"alice", "carol"}

        bob = await repo.find_one(name="bob")
        assert bob is not None
        assert bob.active is False

        ghost = await repo.find_one(name="no_such")
        assert ghost is None
    finally:
        await engine.dispose()


async def test_repo_update_bumps_updated_at(memory_root: MemoryRoot) -> None:
    repo, engine = await _setup_repo(memory_root)
    try:
        u = await repo.add(_DemoUser(name="alice"))
        original_updated = u.updated_at
        original_created = u.created_at

        await asyncio.sleep(0.01)
        u.name = "alice2"
        u.active = False
        updated = await repo.update(u)

        assert updated.name == "alice2"
        assert updated.active is False
        assert updated.updated_at >= original_updated  # bumped
        assert updated.created_at == original_created
    finally:
        await engine.dispose()


async def test_repo_delete_by_instance(memory_root: MemoryRoot) -> None:
    repo, engine = await _setup_repo(memory_root)
    try:
        u = await repo.add(_DemoUser(name="alice"))
        assert await repo.count() == 1

        await repo.delete(u)
        assert await repo.count() == 0
        assert await repo.get_by_id(u.id) is None
    finally:
        await engine.dispose()


async def test_repo_delete_by_id_returns_bool(memory_root: MemoryRoot) -> None:
    repo, engine = await _setup_repo(memory_root)
    try:
        u = await repo.add(_DemoUser(name="alice"))

        assert await repo.delete_by_id(u.id) is True
        assert await repo.delete_by_id(u.id) is False  # already gone
        assert await repo.delete_by_id(99999) is False  # never existed
    finally:
        await engine.dispose()
