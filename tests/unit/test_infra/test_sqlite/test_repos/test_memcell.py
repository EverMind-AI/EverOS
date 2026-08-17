"""Tests for :class:`_MemcellRepo` — bulk primary-key fetch semantics.

Pins the ``find_by_ids`` contract:

- **Chunking**: SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` caps a single
  ``IN (?, ?, ...)`` at 999 params on the shipped amalgamation. The repo
  transparently splits large id lists into ``_SQLITE_IN_CHUNK``-sized
  SELECTs and merges the rows in-memory. The direct-path selector in
  ``extract_user_profile`` can pull an owner's entire episode history
  when ``last_profile_ts=0``, so N is unbounded from the caller side.
- **Order preservation**: rows come back in the caller's list order
  regardless of chunk boundaries — downstream code sorts by timestamp,
  but the pre-sort order is stable across runs.
- **Empty fast-path**: no id list → no SQL, empty result.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos import memcell as memcell_mod
from everos.infra.persistence.sqlite.repos.memcell import _MemcellRepo
from everos.infra.persistence.sqlite.tables import Memcell


@pytest.fixture
async def repo(tmp_path: Path) -> _MemcellRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _MemcellRepo(session_factory=factory)


def _memcell(memcell_id: str) -> Memcell:
    """Build a minimal Memcell row — only ``memcell_id`` uniqueness matters."""
    return Memcell(
        memcell_id=memcell_id,
        app_id="default",
        project_id="default",
        session_id="s_test",
        track="user_memory",
        raw_type="chat",
        message_ids_json="[]",
        sender_ids_json="[]",
        payload_json="{}",
        timestamp=datetime(2026, 5, 17, tzinfo=UTC),
    )


async def test_find_by_ids_empty_returns_empty_without_sql(
    repo: _MemcellRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty input → empty output, no SQL executed."""
    calls = 0
    original = AsyncSession.execute

    async def counting_execute(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", counting_execute)
    rows = await repo.find_by_ids([])
    assert rows == []
    assert calls == 0, "empty id list must not open a session or run SQL"


async def test_find_by_ids_single_chunk_below_cap(repo: _MemcellRepo) -> None:
    """Small id list (below cap) still round-trips correctly and preserves order."""
    real_ids = [f"mc_{i:012d}" for i in range(5)]
    await repo.insert_many([_memcell(mid) for mid in real_ids])

    # Query in shuffled order to prove order-preservation.
    query_order = [real_ids[3], real_ids[0], real_ids[4], real_ids[1], real_ids[2]]
    rows = await repo.find_by_ids(query_order)
    assert [r.memcell_id for r in rows] == query_order


async def test_find_by_ids_handles_chunk_size_larger_than_sqlite_limit(
    repo: _MemcellRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1500 unique ids at chunk=500 → 3 SELECTs, all rows returned in caller order.

    Only every third id points at a real row (500 total); the remaining
    1000 are fake ids exercising the "missing rows are dropped" branch.
    The 500 default chunk cap keeps every SELECT well under SQLite's
    ``SQLITE_MAX_VARIABLE_NUMBER`` (999 on the shipped amalgamation).
    """
    # Materialise 500 real memcells, then interleave 1000 fakes to reach 1500.
    real_ids = [f"mc_real_{i:07d}" for i in range(500)]
    fake_ids = [f"mc_fake_{i:07d}" for i in range(1000)]
    await repo.insert_many([_memcell(mid) for mid in real_ids])

    # Interleave 2 fake per 1 real → 1500 total; every 3rd position is real.
    query_ids: list[str] = []
    for i in range(500):
        query_ids.append(real_ids[i])
        query_ids.append(fake_ids[2 * i])
        query_ids.append(fake_ids[2 * i + 1])
    assert len(query_ids) == 1500

    assert memcell_mod._SQLITE_IN_CHUNK == 500, (
        "test assumes default chunk size; update if the default changes"
    )

    # Count SELECT round-trips against the live AsyncSession.
    selects = 0
    original_execute = AsyncSession.execute

    async def counting_execute(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal selects
        selects += 1
        return await original_execute(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", counting_execute)
    rows = await repo.find_by_ids(query_ids)

    # 1500 ids / chunk 500 = 3 SELECTs.
    assert selects == 3, f"expected 3 chunked SELECTs, got {selects}"

    got_ids = [r.memcell_id for r in rows]
    # De-duplicated (the id list has no duplicates) and only real rows survive.
    assert set(got_ids) == set(real_ids)
    assert len(got_ids) == len(set(got_ids))

    # Order-preserving: real ids come back in the same relative order they
    # appeared in ``query_ids`` (fakes are silently dropped between them).
    expected_order = [mid for mid in query_ids if mid in set(real_ids)]
    assert got_ids == expected_order
