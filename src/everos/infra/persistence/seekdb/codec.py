"""Encode logical records for SeekDB and restore typed result rows.

Datetime values use exact UTC epoch milliseconds, arrays use JSON, and VECTOR
values use textual input on writes while accepting text, native lists, or
little-endian float32 bytes on reads.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from everos.component.utils.datetime import (
    ensure_utc,
    from_timestamp_ms,
    to_timestamp_ms,
)
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
)

from .errors import SeekdbValueLimitError
from .schema import physical_column


def to_row(record: BaseModel, schema: IndexSchema) -> dict[str, Any]:
    """Encode one validated Pydantic record into physical column values."""
    raw = record.model_dump(mode="python")
    out: dict[str, Any] = {}
    for field in schema.fields:
        name, value = write_field_value(field, raw.get(field.name), schema)
        out[name] = value
    return out


def from_row(row: Mapping[str, Any], schema: IndexSchema) -> dict[str, Any]:
    """Restore all logical fields present in a raw SeekDB result row."""
    out: dict[str, Any] = {}
    for field in schema.fields:
        storage_name = _storage_name(field)
        if storage_name not in row:
            continue
        value = row[storage_name]
        if field.kind is IndexFieldKind.DATETIME:
            out[field.name] = None if value is None else from_timestamp_ms(int(value))
        elif field.kind is IndexFieldKind.STRING_ARRAY:
            out[field.name] = _decode_json(value)
        elif field.kind is IndexFieldKind.DENSE_VECTOR:
            out[field.name] = _decode_vector(value, field.dimension)
        else:
            out[field.name] = _decode_scalar(value)
    return out


def model_from_row(
    row: Mapping[str, Any], schema: IndexSchema, model: type[BaseModel]
) -> BaseModel:
    """Restore and validate a complete result row as ``model``."""
    return model.model_validate(from_row(row, schema))


def write_field_value(
    field: IndexField, value: Any, schema: IndexSchema
) -> tuple[str, Any]:
    """Validate and encode one logical field for INSERT or UPDATE."""
    table_name = schema.table_name
    if value is None:
        if not field.nullable:
            raise SeekdbValueLimitError(f"{table_name}.{field.name} cannot be null")
        return _storage_name(field), None
    if field.kind is IndexFieldKind.STRING:
        text = str(value)
        column = physical_column(schema, field.name)
        if limit := _varchar_length(column.sql_type):
            _validate_string_length(table_name, field.name, text, limit)
        return field.name, text
    if field.kind is IndexFieldKind.STRING_ARRAY:
        items = [str(item) for item in value]
        if field.max_capacity is not None and len(items) > field.max_capacity:
            raise SeekdbValueLimitError(
                f"{table_name}.{field.name} has {len(items)} items; "
                f"SeekDB limit is {field.max_capacity}"
            )
        for position, item in enumerate(items):
            if field.max_length is not None:
                _validate_string_length(
                    table_name,
                    f"{field.name}[{position}]",
                    item,
                    field.max_length,
                )
        return field.name, json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    if field.kind is IndexFieldKind.DATETIME:
        return f"{field.name}_ms", _datetime_to_ms(value)
    if field.kind is IndexFieldKind.DENSE_VECTOR:
        vector = _validate_vector(value, field, table_name)
        return field.name, json.dumps(vector, separators=(",", ":"))
    return field.name, value


def _storage_name(field: IndexField) -> str:
    return f"{field.name}_ms" if field.kind is IndexFieldKind.DATETIME else field.name


def _varchar_length(sql_type: str) -> int | None:
    normalized = sql_type.strip().upper()
    if not normalized.startswith("VARCHAR(") or not normalized.endswith(")"):
        return None
    return int(normalized[8:-1])


def _validate_string_length(table: str, field: str, value: str, limit: int) -> None:
    size = len(value)
    if size > limit:
        raise SeekdbValueLimitError(
            f"{table}.{field} is {size} characters; SeekDB limit is {limit}"
        )


def _datetime_to_ms(value: Any) -> int:
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return to_timestamp_ms(aware)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(f"expected datetime or epoch ms, got {type(value).__name__}")


def _validate_vector(
    value: Sequence[float], field: IndexField, table_name: str
) -> list[float]:
    vector = [float(item) for item in value]
    if len(vector) != field.dimension:
        raise SeekdbValueLimitError(
            f"{table_name}.{field.name} has dimension {len(vector)}; "
            f"expected {field.dimension}"
        )
    if not all(math.isfinite(item) for item in vector):
        raise SeekdbValueLimitError(
            f"{table_name}.{field.name} contains a non-finite value"
        )
    return vector


def _decode_scalar(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _decode_json(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value


def _decode_vector(value: Any, dimension: int | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [float(item) for item in json.loads(value)]
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            decoded = ""
        if decoded:
            try:
                parsed = json.loads(decoded)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                if isinstance(parsed, list) and (
                    dimension is None or len(parsed) == dimension
                ):
                    return [float(item) for item in parsed]
        if dimension is not None and len(value) == dimension * 4:
            return list(struct.unpack(f"<{dimension}f", value))
        raise ValueError("unexpected binary VECTOR representation")
    return [float(item) for item in value]


__all__ = ["from_row", "model_from_row", "to_row", "write_field_value"]
