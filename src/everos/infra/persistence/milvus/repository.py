"""Milvus repository for EverOS rebuildable derived indexes."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel
from pymilvus import DataType, Function, FunctionType, MilvusClient

from everos.component.utils.datetime import ensure_utc, from_timestamp, to_timestamp_ms
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.infra.persistence.index.predicate import (
    Predicate,
    all_of,
    eq,
    one_of,
)
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
    schema_for,
)

from .milvus_manager import MilvusSchemaMismatchError, collection_name, get_client
from .predicate import render_predicate

logger = get_logger(__name__)

_DUMMY_VECTOR_FIELD = "_everos_dummy_vector"
_DUMMY_VECTOR_DIMENSION = 2
_SPARSE_SUFFIX = "__sparse"
_PRESENT_SUFFIX = "__present"
_UPDATE_FETCH_LIMIT = 10_000


class MilvusValueLimitError(ValueError):
    """A row exceeds a documented Milvus VARCHAR, array, or vector limit."""


class MilvusRepoBase[T: BaseModel]:
    """Generic Milvus repository backed by one neutral index schema."""

    schema: type[T]
    _write_locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _collection_locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _ready_collections: ClassVar[set[str]] = set()

    @property
    def index_schema(self) -> IndexSchema:
        return schema_for(self.schema)

    @property
    def table_name(self) -> str:
        return self.index_schema.table_name

    @property
    def collection_name(self) -> str:
        return collection_name(self.table_name)

    @classmethod
    def _write_lock(cls, name: str) -> asyncio.Lock:
        return cls._write_locks.setdefault(name, asyncio.Lock())

    @classmethod
    def _collection_lock(cls, name: str) -> asyncio.Lock:
        return cls._collection_locks.setdefault(name, asyncio.Lock())

    @classmethod
    def _reset_collection_cache(cls) -> None:
        cls._ready_collections.clear()
        cls._collection_locks.clear()

    @classmethod
    def _reset_locks_for_tests(cls) -> None:
        cls._write_locks.clear()
        cls._reset_collection_cache()

    async def ensure_collection(self) -> None:
        """Create or verify the collection once per process."""
        name = self.collection_name
        if name in self._ready_collections:
            return
        async with self._collection_lock(name):
            if name in self._ready_collections:
                return
            client = await get_client()
            if await _run(client.has_collection, name):
                await self.verify_collection()
            else:
                await self._create_collection(client)
            self._ready_collections.add(name)

    async def _create_collection(self, client: MilvusClient) -> None:
        schema = self._build_collection_schema()
        index_params = client.prepare_index_params()
        vector_fields = self.index_schema.vector_fields
        if vector_fields:
            for field in vector_fields:
                index_params.add_index(
                    field_name=field.name,
                    index_type="AUTOINDEX",
                    metric_type="COSINE",
                )
        else:
            index_params.add_index(
                field_name=_DUMMY_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
        for field in self.index_schema.bm25_fields:
            index_params.add_index(
                field_name=_sparse_field(field),
                index_type="AUTOINDEX",
                metric_type="BM25",
            )
        settings = load_settings().milvus
        await _run(
            client.create_collection,
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=settings.consistency_level,
        )
        logger.info(
            "milvus_collection_created",
            table=self.table_name,
            collection=self.collection_name,
        )

    async def verify_collection(self) -> None:
        client = await get_client()
        description = await _run(client.describe_collection, self.collection_name)
        actual = {field["name"] for field in description.get("fields", [])}
        expected = set(self._stored_field_names())
        missing = expected - actual
        stale = actual - expected
        if missing or stale:
            raise MilvusSchemaMismatchError(
                f"Milvus collection {self.collection_name!r} schema drift: "
                f"missing={sorted(missing)}, stale={sorted(stale)}. The index is "
                "rebuildable from markdown; run `everos cascade rebuild`."
            )

    async def add(self, records: Sequence[T]) -> None:
        if not records:
            return
        await self.ensure_collection()
        payload = [self._to_milvus_record(record) for record in records]
        client = await get_client()
        async with self._write_lock(self.collection_name):
            await _run(client.insert, self.collection_name, payload)

    async def upsert(self, records: Sequence[T], *, by: str = "id") -> None:
        if by != "id":
            raise ValueError("MilvusRepoBase only supports upsert by id")
        if not records:
            return
        await self.ensure_collection()
        payload = [self._to_milvus_record(record) for record in records]
        client = await get_client()
        async with self._write_lock(self.collection_name):
            await _run(client.upsert, self.collection_name, payload)

    async def update(self, updates: dict[str, Any], *, where: Predicate) -> None:
        rows = await self._query_raw(
            where, limit=_UPDATE_FETCH_LIMIT, include_vectors=True
        )
        if not rows:
            return
        if len(rows) == _UPDATE_FETCH_LIMIT:
            logger.warning(
                "milvus_update_truncated",
                table=self.table_name,
                limit=_UPDATE_FETCH_LIMIT,
            )
        patched: list[dict[str, Any]] = []
        for row in rows:
            merged = dict(row)
            for key, value in updates.items():
                self._write_field_value(merged, key, value)
            self._validate_raw_record(merged)
            patched.append(merged)
        client = await get_client()
        async with self._write_lock(self.collection_name):
            await _run(client.upsert, self.collection_name, patched)

    async def optimize(self, *, cleanup_older_than: dt.timedelta | None = None) -> None:
        """Milvus indexes and compaction are service-managed."""

    async def rebuild_indexes(self) -> None:
        """Milvus AUTOINDEX maintenance is service-managed."""

    async def count(self) -> int:
        return await self._count_where(None)

    async def _count_where(self, where: Predicate | None) -> int:
        await self.ensure_collection()
        client = await get_client()
        rows = await _run(
            client.query,
            self.collection_name,
            filter=self._expr(where),
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> T | None:
        if id_field != "id":
            rows = await self.find_where(eq(id_field, id_value), limit=1)
            return rows[0] if rows else None
        await self.ensure_collection()
        client = await get_client()
        rows = await _run(
            client.get,
            self.collection_name,
            ids=[id_value],
            output_fields=self._output_fields(include_vectors=True),
        )
        return self._model_from_milvus(rows[0]) if rows else None

    async def find_where(self, where: Predicate, *, limit: int = 100) -> list[T]:
        rows = await self._query_raw(where, limit=limit, include_vectors=True)
        return [self._model_from_milvus(row) for row in rows]

    async def find_one_where(self, where: Predicate) -> T | None:
        rows = await self.find_where(where, limit=1)
        return rows[0] if rows else None

    async def find_where_paginated(
        self,
        where: Predicate,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20_000,
    ) -> tuple[list[T], int]:
        total = await self._count_where(where)
        raw = await self._query_raw(where, limit=max_fetch, include_vectors=True)
        if total > len(raw):
            logger.warning(
                "milvus_find_where_paginated_truncated",
                table=self.table_name,
                total=total,
                max_fetch=max_fetch,
            )
        rows = [self._model_from_milvus(row) for row in raw]
        rows.sort(
            key=lambda row: _sort_value(getattr(row, sort_by, None)),
            reverse=descending,
        )
        offset = (page - 1) * page_size
        return rows[offset : offset + page_size], total

    async def find_by_owner(self, owner_id: str, *, limit: int = 100) -> list[T]:
        return await self.find_where(eq("owner_id", owner_id), limit=limit)

    async def find_by_md_path(self, md_path: str) -> T | None:
        return await self.find_one_where(eq("md_path", md_path))

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: Predicate | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if vector is None:
            return await self._query_candidate_rows(where, limit=limit)
        return await self.dense_search(vector, where, limit=limit)

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Predicate | None,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not query_terms:
            return []
        await self.ensure_collection()
        fields = list(columns or self.index_schema.bm25_fields)
        if not fields:
            return []
        unknown = set(fields) - set(self.index_schema.bm25_fields)
        if unknown:
            raise ValueError(f"unknown BM25 fields: {sorted(unknown)}")
        client = await get_client()
        query = " ".join(term for term in query_terms if term)
        best: dict[str, dict[str, Any]] = {}
        for field in fields:
            results = await _run(
                client.search,
                self.collection_name,
                data=[query],
                anns_field=_sparse_field(field),
                filter=self._expr(where),
                limit=limit,
                output_fields=self._output_fields(include_vectors=False),
                search_params={"metric_type": "BM25"},
            )
            for row in _first_result_set(results):
                shaped = self._candidate_row_from_search(row)
                score = _bm25_score_from_distance(row.get("distance"))
                shaped["_score"] = score
                rid = shaped.get("id")
                if not isinstance(rid, str):
                    continue
                prior = best.get(rid)
                if prior is None or score > float(prior.get("_score", 0.0)):
                    best[rid] = shaped
        return sorted(
            best.values(), key=lambda item: float(item.get("_score", 0.0)), reverse=True
        )[:limit]

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Predicate | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        if not vector:
            return []
        field = self.index_schema.field(vector_field)
        if field.kind is not IndexFieldKind.DENSE_VECTOR:
            raise ValueError(f"{vector_field!r} is not a dense-vector field")
        self._validate_vector(field, vector)
        await self.ensure_collection()
        client = await get_client()
        present = eq(_present_field(vector_field), True)
        results = await _run(
            client.search,
            self.collection_name,
            data=[list(vector)],
            anns_field=vector_field,
            filter=self._expr(all_of(where, present)),
            limit=limit,
            output_fields=self._output_fields(include_vectors=False),
            search_params={"metric_type": "COSINE"},
        )
        return [
            self._candidate_row_from_search(row, normalize_cosine=True)
            for row in _first_result_set(results)
        ]

    async def delete(self, predicate: Predicate) -> None:
        await self.ensure_collection()
        client = await get_client()
        async with self._write_lock(self.collection_name):
            await _run(
                client.delete,
                self.collection_name,
                filter=self._expr(predicate),
            )

    async def delete_by_md_path(self, md_path: str) -> int:
        predicate = eq("md_path", md_path)
        count = await self._count_where(predicate)
        if count:
            await self.delete(predicate)
        return count

    async def find_by_owner_entry(
        self,
        owner_id: str,
        entry_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> T | None:
        return await self.find_one_where(
            all_of(
                eq("owner_id", owner_id),
                eq("entry_id", entry_id),
                eq("app_id", app_id),
                eq("project_id", project_id),
            )
        )

    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[T]:
        if not entry_ids:
            return []
        return await self.find_where(
            all_of(
                eq("owner_id", owner_id),
                one_of("entry_id", list(entry_ids)),
                eq("app_id", app_id),
                eq("project_id", project_id),
            ),
            limit=len(entry_ids),
        )

    async def find_by_session(
        self, owner_id: str, session_id: str, *, limit: int = 100
    ) -> list[T]:
        return await self.find_where(
            all_of(eq("owner_id", owner_id), eq("session_id", session_id)),
            limit=limit,
        )

    async def find_by_parent(
        self, parent_type: str, parent_id: str, *, limit: int = 100
    ) -> list[T]:
        return await self.find_where(
            all_of(eq("parent_type", parent_type), eq("parent_id", parent_id)),
            limit=limit,
        )

    def _build_collection_schema(self):  # type: ignore[no-untyped-def]
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for field in self.index_schema.fields:
            self._add_schema_field(schema, field)
        if not self.index_schema.vector_fields:
            schema.add_field(
                field_name=_DUMMY_VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=_DUMMY_VECTOR_DIMENSION,
            )
        for field in self.index_schema.bm25_fields:
            sparse_name = _sparse_field(field)
            schema.add_field(
                field_name=sparse_name,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_function(
                Function(
                    name=f"{field}_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=[field],
                    output_field_names=[sparse_name],
                )
            )
        return schema

    def _add_schema_field(self, schema: Any, field: IndexField) -> None:
        if field.kind is IndexFieldKind.STRING:
            kwargs: dict[str, Any] = {
                "field_name": field.name,
                "datatype": DataType.VARCHAR,
                "max_length": field.max_length,
            }
            if field.primary:
                kwargs["is_primary"] = True
            elif field.nullable and field.name not in self.index_schema.bm25_fields:
                kwargs["nullable"] = True
            if field.name in self.index_schema.bm25_fields:
                kwargs["enable_analyzer"] = True
            schema.add_field(**kwargs)
        elif field.kind is IndexFieldKind.STRING_ARRAY:
            schema.add_field(
                field_name=field.name,
                datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=field.max_capacity,
                max_length=field.max_length,
                nullable=field.nullable,
            )
        elif field.kind is IndexFieldKind.FLOAT:
            schema.add_field(
                field_name=field.name,
                datatype=DataType.DOUBLE,
                nullable=field.nullable,
            )
        elif field.kind is IndexFieldKind.INTEGER:
            schema.add_field(
                field_name=field.name,
                datatype=DataType.INT64,
                nullable=field.nullable,
            )
        elif field.kind is IndexFieldKind.DATETIME:
            schema.add_field(
                field_name=_datetime_storage_field(field.name),
                datatype=DataType.INT64,
                nullable=field.nullable,
            )
        elif field.kind is IndexFieldKind.DENSE_VECTOR:
            schema.add_field(
                field_name=field.name,
                datatype=DataType.FLOAT_VECTOR,
                dim=field.dimension,
            )
            schema.add_field(
                field_name=_present_field(field.name),
                datatype=DataType.BOOL,
            )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise TypeError(f"unsupported index field kind: {field.kind}")

    def _stored_field_names(self) -> list[str]:
        names: list[str] = []
        for field in self.index_schema.fields:
            if field.kind is IndexFieldKind.DATETIME:
                names.append(_datetime_storage_field(field.name))
            else:
                names.append(field.name)
            if field.kind is IndexFieldKind.DENSE_VECTOR:
                names.append(_present_field(field.name))
        if not self.index_schema.vector_fields:
            names.append(_DUMMY_VECTOR_FIELD)
        names.extend(_sparse_field(field) for field in self.index_schema.bm25_fields)
        return names

    def _output_fields(self, *, include_vectors: bool) -> list[str]:
        fields: list[str] = []
        vector_names = {field.name for field in self.index_schema.vector_fields}
        present_names = {_present_field(name) for name in vector_names}
        for name in self._stored_field_names():
            if name.endswith(_SPARSE_SUFFIX):
                continue
            if name == _DUMMY_VECTOR_FIELD:
                if include_vectors:
                    fields.append(name)
                continue
            if not include_vectors and (name in vector_names or name in present_names):
                continue
            fields.append(name)
        return fields

    def _to_milvus_record(self, record: T) -> dict[str, Any]:
        raw = record.model_dump(mode="python")
        out: dict[str, Any] = {}
        for field in self.index_schema.fields:
            value = raw.get(field.name)
            if field.name in self.index_schema.bm25_fields and value is None:
                value = ""
            if field.kind is IndexFieldKind.DATETIME:
                out[_datetime_storage_field(field.name)] = (
                    _datetime_to_ms(value) if value is not None else None
                )
            elif field.kind is IndexFieldKind.DENSE_VECTOR:
                present = value is not None
                out[field.name] = (
                    list(value) if present else [0.0] * int(field.dimension or 0)
                )
                out[_present_field(field.name)] = present
            elif field.kind is IndexFieldKind.STRING_ARRAY:
                out[field.name] = [str(item) for item in (value or [])]
            else:
                out[field.name] = value
        if not self.index_schema.vector_fields:
            out[_DUMMY_VECTOR_FIELD] = [0.0] * _DUMMY_VECTOR_DIMENSION
        self._validate_raw_record(out)
        return out

    def _validate_raw_record(self, row: dict[str, Any]) -> None:
        for field in self.index_schema.fields:
            storage_name = (
                _datetime_storage_field(field.name)
                if field.kind is IndexFieldKind.DATETIME
                else field.name
            )
            value = row.get(storage_name)
            if value is None:
                if not field.nullable:
                    raise MilvusValueLimitError(
                        f"{self.table_name}.{field.name} cannot be null"
                    )
                continue
            if field.kind is IndexFieldKind.STRING:
                size = len(str(value).encode("utf-8"))
                if field.max_length is not None and size > field.max_length:
                    raise MilvusValueLimitError(
                        f"{self.table_name}.{field.name} is {size} UTF-8 bytes; "
                        f"Milvus limit is {field.max_length}"
                    )
            elif field.kind is IndexFieldKind.STRING_ARRAY:
                if field.max_capacity is not None and len(value) > field.max_capacity:
                    raise MilvusValueLimitError(
                        f"{self.table_name}.{field.name} has {len(value)} items; "
                        f"Milvus limit is {field.max_capacity}"
                    )
                for position, item in enumerate(value):
                    size = len(str(item).encode("utf-8"))
                    if field.max_length is not None and size > field.max_length:
                        raise MilvusValueLimitError(
                            f"{self.table_name}.{field.name}[{position}] is {size} "
                            f"UTF-8 bytes; Milvus limit is {field.max_length}"
                        )
            elif field.kind is IndexFieldKind.DENSE_VECTOR:
                self._validate_vector(field, value)

    def _validate_vector(self, field: IndexField, value: Sequence[float]) -> None:
        if len(value) != field.dimension:
            raise MilvusValueLimitError(
                f"{self.table_name}.{field.name} has dimension {len(value)}; "
                f"expected {field.dimension}"
            )

    def _model_from_milvus(self, row: dict[str, Any]) -> T:
        return self.schema.model_validate(self._restore_row(row))

    def _candidate_row_from_search(
        self, row: dict[str, Any], *, normalize_cosine: bool = False
    ) -> dict[str, Any]:
        shaped = self._restore_row(row.get("entity", {}))
        raw_distance = row.get("distance")
        shaped["_distance"] = (
            _cosine_distance_from_milvus(raw_distance)
            if normalize_cosine
            else raw_distance
        )
        return shaped

    def _restore_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field in self.index_schema.fields:
            if field.kind is IndexFieldKind.DATETIME:
                value = row.get(_datetime_storage_field(field.name))
                out[field.name] = None if value is None else from_timestamp(int(value))
            elif field.kind is IndexFieldKind.DENSE_VECTOR:
                if field.name in row:
                    out[field.name] = (
                        row[field.name]
                        if row.get(_present_field(field.name), True)
                        else None
                    )
            elif field.name in row:
                value = row[field.name]
                if (
                    field.name in self.index_schema.bm25_fields
                    and value == ""
                    and field.nullable
                ):
                    value = None
                out[field.name] = value
        return out

    def _write_field_value(
        self, row: dict[str, Any], field_name: str, value: Any
    ) -> None:
        field = self.index_schema.field(field_name)
        if field.kind is IndexFieldKind.DATETIME:
            row[_datetime_storage_field(field_name)] = (
                _datetime_to_ms(value) if value is not None else None
            )
        elif field.kind is IndexFieldKind.DENSE_VECTOR:
            present = value is not None
            row[field_name] = (
                list(value) if present else [0.0] * int(field.dimension or 0)
            )
            row[_present_field(field_name)] = present
        else:
            row[field_name] = value

    async def _query_raw(
        self,
        where: Predicate | None,
        *,
        limit: int,
        include_vectors: bool,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection()
        client = await get_client()
        return await _run(
            client.query,
            self.collection_name,
            filter=self._expr(where),
            output_fields=self._output_fields(include_vectors=include_vectors),
            limit=limit,
        )

    async def _query_candidate_rows(
        self, where: Predicate | None, *, limit: int
    ) -> list[dict[str, Any]]:
        rows = await self._query_raw(where, limit=limit, include_vectors=False)
        return [self._restore_row(row) for row in rows]

    def _expr(self, where: Predicate | None) -> str:
        if where is not None and not isinstance(where, Predicate):
            raise TypeError(
                "Milvus repository predicates must use the neutral Predicate AST, "
                f"got {type(where).__name__}"
            )
        return render_predicate(
            where,
            datetime_fields=self.index_schema.datetime_fields,
        )


def _datetime_storage_field(name: str) -> str:
    return f"{name}_ms"


def _present_field(name: str) -> str:
    return f"{name}{_PRESENT_SUFFIX}"


def _sparse_field(name: str) -> str:
    return f"{name}{_SPARSE_SUFFIX}"


def _datetime_to_ms(value: Any) -> int:
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return to_timestamp_ms(aware)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(f"expected datetime or epoch ms, got {type(value).__name__}")


def _sort_value(value: Any) -> Any:
    if value is None:
        fallback = ensure_utc(dt.datetime.min)
        assert fallback is not None
        return fallback
    return value


def _bm25_score_from_distance(distance: Any) -> float:
    """Milvus BM25 is higher-is-better; expose a non-negative score."""
    return 0.0 if distance is None else max(0.0, float(distance))


def _cosine_distance_from_milvus(distance: Any) -> float | None:
    """Convert Milvus Server / Zilliz similarity to Lance-style distance."""
    if distance is None:
        return None
    return min(1.0, max(0.0, 1.0 - float(distance)))


def _first_result_set(results: Any) -> list[dict[str, Any]]:
    if not results:
        return []
    return list(results[0] or [])


async def _run(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)


__all__ = ["MilvusRepoBase", "MilvusValueLimitError"]
