"""Search reads leave the vector columns behind unless asked for them.

Turning a result into Python objects runs on the event loop; with two
1024-float vectors per row it was most of the loop's busy time under a
search-heavy soak (2026-11), and every search caller dropped the vectors
straight after. ``with_vectors=True`` keeps them for the callers that need
them (the vector and cluster backfills).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from pathlib import Path

import lancedb
import pytest

from everos.infra.persistence.lancedb._base.indexes import ensure_fts_indexes
from everos.infra.persistence.lancedb._repos.agent_case import _AgentCaseRepo
from everos.infra.persistence.records import AgentCase
from tests.fakes import arrow_schema

_DIM = 8


def _case(i: int) -> AgentCase:
    return AgentCase(
        id=f"a1_ac_{i}",
        entry_id=f"ac_{i}",
        owner_id="a1",
        owner_type="agent",
        session_id="s1",
        timestamp=dt.datetime(2026, 11, 5, 10, i, tzinfo=dt.UTC),
        parent_id=f"mc_{i}",
        quality_score=0.5,
        task_intent=f"plan hiking trip number {i}",
        task_intent_tokens=f"plan hiking trip number {i}",
        approach="check the weather",
        approach_tokens="check the weather",
        md_path="agents/a1/.cases/ac.md",
        content_sha256=f"sha{i}",
        vector=[0.1 * (i + 1)] * _DIM,
    )


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncIterator[_AgentCaseRepo]:
    conn = await lancedb.connect_async(str(tmp_path / "lancedb"))
    table = await conn.create_table(
        AgentCase.TABLE_NAME, schema=arrow_schema(AgentCase, dim=_DIM)
    )
    r = _AgentCaseRepo(table=table, dim=_DIM)
    await r.upsert([_case(i) for i in range(5)])
    await ensure_fts_indexes(table, r.index_schema)
    yield r


async def test_dense_search_leaves_vectors_behind_unless_asked(
    repo: _AgentCaseRepo,
) -> None:
    query = [0.3] * _DIM
    bare = await repo.dense_search(query, None, limit=3)
    full = await repo.dense_search(query, None, limit=3, with_vectors=True)

    assert bare and all("vector" not in row for row in bare)
    assert all(len(row["vector"]) == _DIM for row in full)
    assert [(r["id"], r["_distance"]) for r in bare] == [
        (r["id"], r["_distance"]) for r in full
    ]


async def test_sparse_search_leaves_vectors_behind_unless_asked(
    repo: _AgentCaseRepo,
) -> None:
    bare = await repo.sparse_search(["hiking"], None, limit=5)
    full = await repo.sparse_search(["hiking"], None, limit=5, with_vectors=True)

    assert bare and all("vector" not in row for row in bare)
    assert all(row["vector"] is not None for row in full)
    assert [(r["id"], r["_score"]) for r in bare] == [
        (r["id"], r["_score"]) for r in full
    ]


async def test_scan_leaves_vectors_behind_unless_asked(repo: _AgentCaseRepo) -> None:
    bare = await repo.scan()
    full = await repo.scan(with_vectors=True)

    assert sorted(f.id for f in bare) == sorted(f.id for f in full)
    assert all(f.vector is None for f in bare)
    assert all(f.vector is not None for f in full)
