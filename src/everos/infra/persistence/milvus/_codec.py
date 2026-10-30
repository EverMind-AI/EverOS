"""Translation between EverOS records and Milvus rows, with Milvus' value limits.

:class:`RowCodec` turns one pydantic record type into the physical row a
collection stores (datetimes as epoch ms, null vectors as a zero vector plus a
presence flag, the joined ``__pk``) and back, and rejects a value Milvus would
refuse — before the write, with the field named — as
:class:`MilvusValueLimitError`. Pure: no client, no IO.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from everos.core.utils.datetime import (
    ensure_utc,
    from_timestamp_ms,
    to_timestamp_ms,
)
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
)

from ._columns import (
    DUMMY_VECTOR_DIMENSION,
    DUMMY_VECTOR_FIELD,
    PK_FIELD,
    PK_MAX_LENGTH,
    PK_SEPARATOR,
    datetime_storage_field,
    present_field,
)


class MilvusValueLimitError(ValueError):
    """A row exceeds a documented Milvus VARCHAR, array, or vector limit."""


def compose_pk(schema: IndexSchema, row: Mapping[str, Any]) -> str:
    """Join the row's PRIMARY_KEY values into the ``__pk`` scalar.

    Milvus keys a collection on exactly one field, so the declared key
    columns are concatenated in declaration order. Rejects a component
    containing the separator (the join would stop being injective) and a
    result over the VARCHAR limit (:meth:`RowCodec.validate_raw` only covers
    logical fields, not this physical one).
    """
    parts = [str(row[column]) for column in schema.primary_key]
    for column, part in zip(schema.primary_key, parts, strict=True):
        if PK_SEPARATOR in part:
            raise MilvusValueLimitError(
                f"{schema.table_name}.{column} contains {PK_SEPARATOR!r}: {part!r}"
            )
    pk = PK_SEPARATOR.join(parts)
    size = len(pk.encode("utf-8"))
    if size > PK_MAX_LENGTH:
        raise MilvusValueLimitError(
            f"{schema.table_name}.{PK_FIELD} is {size} UTF-8 bytes; "
            f"Milvus limit is {PK_MAX_LENGTH}"
        )
    return pk


class RowCodec[T: BaseModel]:
    """Encodes one record type as Milvus rows and decodes rows back.

    Args:
        model: The pydantic record type the collection stores.
        index_schema: That type's neutral index schema.
    """

    def __init__(self, model: type[T], index_schema: IndexSchema) -> None:
        self._model = model
        self._index_schema = index_schema
        self._table_name = index_schema.table_name

    def to_record(self, record: T) -> dict[str, Any]:
        raw = record.model_dump(mode="python")
        out: dict[str, Any] = {}
        for field in self._index_schema.fields:
            value = raw.get(field.name)
            if field.name in self._index_schema.bm25_fields and value is None:
                value = ""
            if field.kind is IndexFieldKind.DATETIME:
                out[datetime_storage_field(field.name)] = (
                    _datetime_to_ms(value) if value is not None else None
                )
            elif field.kind is IndexFieldKind.DENSE_VECTOR:
                present = value is not None
                out[field.name] = (
                    list(value) if present else [0.0] * int(field.dimension or 0)
                )
                out[present_field(field.name)] = present
            elif field.kind is IndexFieldKind.STRING_ARRAY:
                out[field.name] = [str(item) for item in (value or [])]
            else:
                out[field.name] = value
        if not self._index_schema.vector_fields:
            out[DUMMY_VECTOR_FIELD] = [0.0] * DUMMY_VECTOR_DIMENSION
        out[PK_FIELD] = compose_pk(self._index_schema, out)
        self.validate_raw(out)
        return out

    def validate_raw(self, row: dict[str, Any]) -> None:
        for field in self._index_schema.fields:
            storage_name = (
                datetime_storage_field(field.name)
                if field.kind is IndexFieldKind.DATETIME
                else field.name
            )
            value = row.get(storage_name)
            if value is None:
                if not field.nullable:
                    raise MilvusValueLimitError(
                        f"{self._table_name}.{field.name} cannot be null"
                    )
                continue
            if field.kind is IndexFieldKind.STRING:
                size = len(str(value).encode("utf-8"))
                if field.max_length is not None and size > field.max_length:
                    raise MilvusValueLimitError(
                        f"{self._table_name}.{field.name} is {size} UTF-8 bytes; "
                        f"Milvus limit is {field.max_length}"
                    )
            elif field.kind is IndexFieldKind.STRING_ARRAY:
                if field.max_capacity is not None and len(value) > field.max_capacity:
                    raise MilvusValueLimitError(
                        f"{self._table_name}.{field.name} has {len(value)} items; "
                        f"Milvus limit is {field.max_capacity}"
                    )
                for position, item in enumerate(value):
                    size = len(str(item).encode("utf-8"))
                    if field.max_length is not None and size > field.max_length:
                        raise MilvusValueLimitError(
                            f"{self._table_name}.{field.name}[{position}] is {size} "
                            f"UTF-8 bytes; Milvus limit is {field.max_length}"
                        )
            elif field.kind is IndexFieldKind.DENSE_VECTOR:
                self.validate_vector(field, value)

    def validate_vector(self, field: IndexField, value: Sequence[float]) -> None:
        if len(value) != field.dimension:
            raise MilvusValueLimitError(
                f"{self._table_name}.{field.name} has dimension {len(value)}; "
                f"expected {field.dimension}"
            )

    def to_model(self, row: dict[str, Any]) -> T:
        return self._model.model_validate(self.restore_row(row))

    def candidate_from_search(
        self, row: dict[str, Any], *, normalize_cosine: bool = False
    ) -> dict[str, Any]:
        shaped = self.restore_row(row.get("entity", {}))
        raw_distance = row.get("distance")
        shaped["_distance"] = (
            cosine_distance_from_milvus(raw_distance)
            if normalize_cosine
            else raw_distance
        )
        return shaped

    def restore_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field in self._index_schema.fields:
            if field.kind is IndexFieldKind.DATETIME:
                value = row.get(datetime_storage_field(field.name))
                out[field.name] = (
                    None if value is None else from_timestamp_ms(int(value))
                )
            elif field.kind is IndexFieldKind.DENSE_VECTOR:
                if field.name in row:
                    out[field.name] = (
                        row[field.name]
                        if row.get(present_field(field.name), True)
                        else None
                    )
            elif field.name in row:
                value = row[field.name]
                if (
                    field.name in self._index_schema.bm25_fields
                    and value == ""
                    and field.nullable
                ):
                    value = None
                out[field.name] = value
        return out

    def write_field_value(
        self, row: dict[str, Any], field_name: str, value: Any
    ) -> None:
        field = self._index_schema.field(field_name)
        if field.kind is IndexFieldKind.DATETIME:
            row[datetime_storage_field(field_name)] = (
                _datetime_to_ms(value) if value is not None else None
            )
        elif field.kind is IndexFieldKind.DENSE_VECTOR:
            present = value is not None
            row[field_name] = (
                list(value) if present else [0.0] * int(field.dimension or 0)
            )
            row[present_field(field_name)] = present
        else:
            row[field_name] = value


def _datetime_to_ms(value: Any) -> int:
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return to_timestamp_ms(aware)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(f"expected datetime or epoch ms, got {type(value).__name__}")


def bm25_score_from_distance(distance: Any) -> float:
    """Milvus BM25 is higher-is-better; expose a non-negative score."""
    return 0.0 if distance is None else max(0.0, float(distance))


def cosine_distance_from_milvus(distance: Any) -> float | None:
    """Convert Milvus Server / Zilliz similarity to Lance-style distance."""
    if distance is None:
        return None
    return min(1.0, max(0.0, 1.0 - float(distance)))
