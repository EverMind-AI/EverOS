"""``ensure_vector_indexes`` — the ANN index on vector columns.

Without it every ``nearest_to`` is a brute-force scan of the column, linear
in rows; the soak measured 0.6 s per scan at 27k rows of 1024-dim vectors.
The index is built only once a column holds ``min_rows`` non-null vectors,
so a Tier 1 store (no embeddings) and a small store are left alone.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar

import lancedb
import pytest
from lancedb import AsyncTable

from everos.infra.persistence.index import schema_for
from everos.infra.persistence.lancedb._base import indexes
from everos.infra.persistence.lancedb._base.indexes import (
    VECTOR_QUERY_NPROBES,
    ensure_vector_indexes,
)
from everos.infra.persistence.records import DenseVector, IndexRecord
from tests.fakes import arrow_schema

_DIM = 8


class _VecSpec(IndexRecord):
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
    TABLE_NAME: ClassVar[str] = "vec_probe"
    BM25_FIELDS: ClassVar[list[str]] = ["body"]

    id: str
    body: str
    vector: DenseVector | None = None  # type: ignore[valid-type]


@pytest.fixture
async def vec_table(tmp_path: Path) -> AsyncIterator[AsyncTable]:
    conn = await lancedb.connect_async(str(tmp_path / "lancedb"))
    yield await conn.create_table(
        _VecSpec.TABLE_NAME, schema=arrow_schema(_VecSpec, dim=_DIM)
    )


def _rows(n: int, *, with_vectors: bool = True) -> list[_VecSpec]:
    rng = random.Random(7)
    return [
        _VecSpec(
            id=f"r{i}",
            body=f"row {i}",
            vector=[rng.random() for _ in range(_DIM)] if with_vectors else None,
        )
        for i in range(n)
    ]


def _dumped(n: int, *, with_vectors: bool = True) -> list[dict[str, object]]:
    return [row.model_dump() for row in _rows(n, with_vectors=with_vectors)]


async def _vector_indices(table: AsyncTable) -> list[tuple[str, str]]:
    return [
        (idx.columns[0], idx.index_type)
        for idx in await table.list_indices()
        if idx.columns and idx.columns[0] == "vector"
    ]


def test_vector_columns_are_the_dense_vector_fields() -> None:
    assert [f.name for f in schema_for(_VecSpec, dim=_DIM).vector_fields] == ["vector"]


async def test_below_the_row_threshold_no_index_is_built(
    vec_table: AsyncTable,
) -> None:
    await vec_table.add(_dumped(40))
    assert (
        await ensure_vector_indexes(
            vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
        )
        == []
    )
    assert await _vector_indices(vec_table) == []


async def test_at_the_threshold_an_ivf_flat_index_is_built(
    vec_table: AsyncTable,
) -> None:
    await vec_table.add(_dumped(60))
    built = await ensure_vector_indexes(
        vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
    )
    assert built == ["vector"]
    assert await _vector_indices(vec_table) == [("vector", "IvfFlat")]


async def _num_indices(table: AsyncTable) -> int:
    (name,) = [
        idx.name
        for idx in await table.list_indices()
        if idx.columns and idx.columns[0] == "vector"
    ]
    stats = await table.index_stats(name)
    assert stats is not None
    return stats.num_indices


async def test_delta_indexes_left_by_optimize_are_collapsed_past_the_cap(
    vec_table: AsyncTable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ``optimize()`` on a table with new rows appends a delta index and
    a query probes them all. Up to the cap they are tolerated (a retrain
    rewrites the whole index); past it the heavy beat folds them into one."""
    monkeypatch.setattr(indexes, "VECTOR_INDEX_MAX_DELTAS", 2)
    await vec_table.add(_dumped(60))
    assert await ensure_vector_indexes(
        vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
    ) == ["vector"]
    assert (
        await ensure_vector_indexes(
            vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
        )
        == []
    )
    await vec_table.add(_dumped(5))
    await vec_table.optimize()
    assert await _num_indices(vec_table) == 2, "precondition: one delta per beat"
    assert (
        await ensure_vector_indexes(
            vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
        )
        == []
    ), "within the cap the index is left alone"
    await vec_table.add(_dumped(5))
    await vec_table.optimize()
    assert await _num_indices(vec_table) == 3
    assert await ensure_vector_indexes(
        vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
    ) == ["vector"]
    assert await _num_indices(vec_table) == 1
    assert len(await _vector_indices(vec_table)) == 1


async def test_nprobes_is_accepted_on_an_unindexed_column(
    vec_table: AsyncTable,
) -> None:
    """``dense_search`` always sets ``nprobes``; a table below the index
    threshold (Tier 1, a fresh store) must still answer."""
    await vec_table.add(_dumped(10))
    rows = await (
        vec_table.query()
        .nearest_to(_rows(1)[0].vector)
        .column("vector")
        .distance_type("cosine")
        .nprobes(VECTOR_QUERY_NPROBES)
        .limit(3)
        .to_list()
    )
    assert len(rows) == 3


async def test_all_null_vectors_are_skipped_even_above_the_threshold(
    vec_table: AsyncTable,
) -> None:
    """A Tier 1 store has rows but no embeddings; there is nothing to train."""
    await vec_table.add(_dumped(60, with_vectors=False))
    assert (
        await ensure_vector_indexes(
            vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50
        )
        == []
    )
    assert await _vector_indices(vec_table) == []


async def test_indexed_search_returns_the_same_neighbours_as_the_scan(
    vec_table: AsyncTable,
) -> None:
    """IVF_FLAT keeps exact distances inside the probed partitions: at this
    size the top-5 must match the brute-force answer."""
    rows = _rows(300)
    await vec_table.add([row.model_dump() for row in rows])
    q = rows[17].vector

    async def _top5(bypass: bool) -> list[str]:
        query = (
            vec_table.query()
            .nearest_to(q)
            .column("vector")
            .distance_type("cosine")
            .limit(5)
        )
        if bypass:
            query = query.bypass_vector_index()
        return [r["id"] for r in await query.to_list()]

    flat = await _top5(bypass=True)
    await ensure_vector_indexes(vec_table, schema_for(_VecSpec, dim=_DIM), min_rows=50)
    indexed = await _top5(bypass=False)
    assert indexed[0] == "r17"
    assert set(indexed) == set(flat)
