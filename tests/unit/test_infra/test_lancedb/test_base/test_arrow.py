"""``arrow_schema_for``: the neutral index schema -> LanceDB's Arrow schema.

A pure function: one case per neutral kind, the nullability carried over, and
a kind without a mapping refused instead of guessed. Plus the write-side
guard that goes with it: a vector whose length is not the configured width is
refused as a named configuration error before Arrow sees it.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pytest

from everos.config import LanceDBSettings
from everos.core.errors import IndexSchemaMismatchError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
)
from everos.infra.persistence.lancedb import LanceRepoBase, open_lancedb_connection
from everos.infra.persistence.lancedb._base import arrow_schema_for
from everos.infra.persistence.records import DenseVector, IndexRecord
from tests.fakes import arrow_schema


def _schema(*fields: IndexField) -> IndexSchema:
    return IndexSchema("_t", object, fields, (), ("id",))


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (IndexField("s", IndexFieldKind.STRING), pa.string()),
        (IndexField("a", IndexFieldKind.STRING_ARRAY), pa.list_(pa.string())),
        (IndexField("f", IndexFieldKind.FLOAT), pa.float64()),
        (IndexField("i", IndexFieldKind.INTEGER), pa.int64()),
        (IndexField("t", IndexFieldKind.DATETIME), pa.timestamp("us", tz="UTC")),
        (
            IndexField("v", IndexFieldKind.DENSE_VECTOR, dimension=7),
            pa.list_(pa.float32(), 7),
        ),
    ],
    ids=lambda v: v.kind.value if isinstance(v, IndexField) else "",
)
def test_each_kind_maps_to_its_arrow_type(
    field: IndexField, expected: pa.DataType
) -> None:
    assert arrow_schema_for(_schema(field)).field(field.name).type == expected


@pytest.mark.parametrize("nullable", [False, True])
def test_nullability_is_carried_over(nullable: bool) -> None:
    field = IndexField("s", IndexFieldKind.STRING, nullable=nullable)
    assert arrow_schema_for(_schema(field)).field("s").nullable is nullable


def test_field_order_is_the_declaration_order() -> None:
    fields = (
        IndexField("b", IndexFieldKind.STRING),
        IndexField("a", IndexFieldKind.INTEGER),
    )
    assert arrow_schema_for(_schema(*fields)).names == ["b", "a"]


def test_a_vector_without_a_width_is_refused() -> None:
    field = IndexField("v", IndexFieldKind.DENSE_VECTOR)
    with pytest.raises(ValueError, match="no dimension"):
        arrow_schema_for(_schema(field))


def test_an_unknown_kind_is_refused() -> None:
    field = IndexField("x", "sparse_vector")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no LanceDB column type"):
        arrow_schema_for(_schema(field))


class _Vec(IndexRecord):
    TABLE_NAME: ClassVar[str] = "_vec"
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
    id: str
    vector: DenseVector | None = None


class _VecRepo(LanceRepoBase[_Vec]):
    schema = _Vec


async def test_a_wrong_width_vector_is_a_named_configuration_error(
    tmp_path: Path,
) -> None:
    LanceRepoBase._reset_locks_for_tests()
    root = MemoryRoot(tmp_path)
    root.ensure()
    conn = await open_lancedb_connection(root.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("_vec", schema=arrow_schema(_Vec, dim=4))
    repo = _VecRepo(table=table, dim=4)

    await repo.upsert([_Vec(id="ok", vector=[0.1] * 4), _Vec(id="none")])
    with pytest.raises(IndexSchemaMismatchError) as excinfo:
        await repo.upsert([_Vec(id="bad", vector=[0.1] * 5)])

    message = str(excinfo.value)
    for part in ("'_vec'", "'vector'", "width 4", "got 5"):
        assert part in message
    assert await repo.count() == 2, "nothing of the refused batch is written"
