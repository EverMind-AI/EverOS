"""The Arrow schema of a LanceDB table, derived from the neutral index schema.

LanceDB stores Arrow, so this is where the backend-neutral
:class:`~everos.infra.persistence.index.schema.IndexSchema` becomes a physical
layout. The mapping is fixed and total: a kind it does not know raises, so a
new kind added to the neutral vocabulary cannot silently create a column of
the wrong type.

Every datetime column is ``timestamp[us, tz=UTC]``: PyArrow ``astimezone``-s
aware input to UTC on write and returns aware UTC datetimes on read, so no
caller converts by hand and storage is always UTC (the two-zone discipline in
``docs/datetime.md``).
"""

from __future__ import annotations

__all__ = ["arrow_schema_for"]

import pyarrow as pa

from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
)

_SCALAR_TYPES: dict[IndexFieldKind, pa.DataType] = {
    IndexFieldKind.STRING: pa.string(),
    IndexFieldKind.STRING_ARRAY: pa.list_(pa.string()),
    IndexFieldKind.FLOAT: pa.float64(),
    IndexFieldKind.INTEGER: pa.int64(),
    IndexFieldKind.DATETIME: pa.timestamp("us", tz="UTC"),
}


def arrow_schema_for(index_schema: IndexSchema) -> pa.Schema:
    """The Arrow schema a LanceDB table of ``index_schema`` is created with."""
    return pa.schema(
        pa.field(field.name, _arrow_type(field), nullable=field.nullable)
        for field in index_schema.fields
    )


def _arrow_type(field: IndexField) -> pa.DataType:
    if field.kind is IndexFieldKind.DENSE_VECTOR:
        if field.dimension is None:
            raise ValueError(f"vector field {field.name!r} has no dimension")
        return pa.list_(pa.float32(), field.dimension)
    try:
        return _SCALAR_TYPES[field.kind]
    except KeyError:
        raise ValueError(
            f"field {field.name!r} has kind {field.kind!r}, which has no "
            f"LanceDB column type"
        ) from None
