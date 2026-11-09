"""Search results omit vectors before Arrow and Python materialization.

``with_vectors=True`` retains them for callers such as vector backfills.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest
from lancedb.query import AsyncQueryBase

from everos.infra.persistence.lancedb._base.indexes import ensure_fts_indexes
from everos.infra.persistence.lancedb._base.repository import LanceRepoBase
from everos.infra.persistence.lancedb._repos.atomic_fact import _AtomicFactRepo
from everos.infra.persistence.lancedb._repos.episode import _EpisodeRepo
from everos.infra.persistence.records import AtomicFact, Episode
from tests.fakes import arrow_schema

_DIM = 8


def _fact(i: int) -> AtomicFact:
    return AtomicFact(
        id=f"u1_af_{i}",
        entry_id=f"af_{i}",
        owner_id="u1",
        owner_type="user",
        timestamp=dt.datetime(2026, 11, 5, 10, i, tzinfo=dt.UTC),
        parent_id=f"ep_{i}",
        sender_ids=["u1"],
        fact=f"likes hiking number {i}",
        fact_tokens=f"likes hiking number {i}",
        md_path="users/u1/.atomic_facts/af.md",
        content_sha256=f"sha{i}",
        vector=[0.1 * (i + 1)] * _DIM,
    )


@pytest.fixture
async def repo(tmp_path: Path) -> AsyncIterator[_AtomicFactRepo]:
    conn = await lancedb.connect_async(str(tmp_path / "lancedb"))
    table = await conn.create_table(
        AtomicFact.TABLE_NAME, schema=arrow_schema(AtomicFact, dim=_DIM)
    )
    r = _AtomicFactRepo(table=table, dim=_DIM)
    await r.upsert([_fact(i) for i in range(5)])
    await ensure_fts_indexes(table, r.index_schema)
    yield r


async def test_dense_search_leaves_vectors_behind_unless_asked(
    repo: _AtomicFactRepo,
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
    repo: _AtomicFactRepo,
) -> None:
    bare = await repo.sparse_search(["hiking"], None, limit=5)
    full = await repo.sparse_search(["hiking"], None, limit=5, with_vectors=True)

    assert bare and all("vector" not in row for row in bare)
    assert all(row["vector"] is not None for row in full)
    assert [(r["id"], r["_score"]) for r in bare] == [
        (r["id"], r["_score"]) for r in full
    ]


async def test_scan_leaves_vectors_behind_unless_asked(repo: _AtomicFactRepo) -> None:
    bare = await repo.scan()
    full = await repo.scan(with_vectors=True)

    assert sorted(f.id for f in bare) == sorted(f.id for f in full)
    assert all(f.vector is None for f in bare)
    assert all(f.vector is not None for f in full)


@pytest.mark.parametrize(
    "method", ["base_search", "dense", "subject", "sparse", "scan"]
)
async def test_query_result_omits_vectors_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Observe real database results, before the repository can drop columns.

    Both vectors must be absent from the Arrow result, with the same filtered
    rows, scores and non-vector content as a full query. Distinct scores avoid
    depending on LanceDB's unspecified ordering among ties.
    """
    conn = await lancedb.connect_async(str(tmp_path / "lancedb"))
    table = await conn.create_table(
        Episode.TABLE_NAME, schema=arrow_schema(Episode, dim=_DIM)
    )
    repo = _EpisodeRepo(table=table, dim=_DIM)
    records = [
        Episode(
            id=f"ep_{i}",
            entry_id=f"ep_{i}",
            owner_id="u1" if i < 3 else "other",
            owner_type="user",
            timestamp=dt.datetime(2026, 11, 9, tzinfo=dt.UTC),
            parent_id=f"mc_{i}",
            sender_ids=["u1"],
            episode=f"hiking {'camping ' * i}",
            episode_tokens=f"hiking {'camping ' * i}",
            md_path=f"users/u1/episodes/ep_{i}.md",
            content_sha256=f"sha{i}",
            vector=[1.0, float(i), *([0.0] * (_DIM - 2))],
            subject_vector=[1.0, float(i * 2), *([0.0] * (_DIM - 2))],
        )
        for i in range(4)
    ]
    await repo.upsert(records)
    await ensure_fts_indexes(table, repo.index_schema)
    observed: list[pa.Table] = []
    original = AsyncQueryBase.to_arrow

    async def observe(query: AsyncQueryBase, *args, **kwargs) -> pa.Table:
        result = await original(query, *args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(AsyncQueryBase, "to_arrow", observe)
    where = "owner_id = 'u1'"
    vector = [1.0, *([0.0] * (_DIM - 1))]

    async def read(with_vectors: bool) -> list[dict]:
        kwargs = {"where": where, "with_vectors": with_vectors}
        if method == "scan":
            rows = await repo.scan(**kwargs)
            return [row.model_dump() for row in rows]
        if method == "base_search":
            return await LanceRepoBase.search(
                repo,
                limit=2,
                **kwargs,
            )
        if method == "sparse":
            return await repo.sparse_search(["hiking"], limit=2, **kwargs)
        return await repo.dense_search(
            vector,
            vector_field="subject_vector" if method == "subject" else "vector",
            limit=2,
            **kwargs,
        )

    full = await read(True)
    vectors = {"vector", "subject_vector"}
    assert observed and all(vectors <= set(t.column_names) for t in observed)
    assert all(row["vector"] and row["subject_vector"] for row in full)
    observed.clear()
    bare = await read(False)
    assert len(bare) == (3 if method == "scan" else 2)
    assert all(row["owner_id"] == "u1" for row in bare)
    if method in {"dense", "subject", "sparse"}:
        assert [row["id"] for row in bare] == ["ep_0", "ep_1"]
        score = "_score" if method == "sparse" else "_distance"
        assert bare[0][score] != bare[1][score]

    def content(rows: list[dict]) -> list[dict]:
        return sorted(
            [{k: v for k, v in row.items() if k not in vectors} for row in rows],
            key=lambda row: row["id"],
        )

    assert content(bare) == content(full)
    assert observed and all(not vectors.intersection(t.column_names) for t in observed)


async def test_base_vector_query_projects_results(
    repo: _AtomicFactRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generic ANN API infers its column on single-vector schemas."""
    original = AsyncQueryBase.to_arrow
    observed: list[pa.Table] = []

    async def observe(query: AsyncQueryBase, *args, **kwargs) -> pa.Table:
        result = await original(query, *args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(AsyncQueryBase, "to_arrow", observe)
    rows = await LanceRepoBase.search(repo, vector=[0.0] * _DIM, limit=2)
    assert [row["id"] for row in rows] == ["u1_af_0", "u1_af_1"]
    assert rows[0]["_distance"] < rows[1]["_distance"]
    assert observed and all("vector" not in t.column_names for t in observed)
