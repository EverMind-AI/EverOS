"""Milvus repository for EverOS rebuildable derived indexes."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel
from pymilvus import DataType, Function, FunctionType, MilvusClient

from everos.component.utils.datetime import (
    ensure_utc,
    from_timestamp_ms,
    to_timestamp_ms,
)
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
    schema_for,
)
from everos.infra.persistence.predicate import (
    Predicate,
    all_of,
    eq,
    one_of,
)

from .milvus_manager import MilvusSchemaMismatchError, collection_name, get_client
from .predicate import render_predicate

logger = get_logger(__name__)

_DUMMY_VECTOR_FIELD = "_everos_dummy_vector"
_DUMMY_VECTOR_DIMENSION = 2
_SPARSE_SUFFIX = "__sparse"
_PRESENT_SUFFIX = "__present"


class MilvusValueLimitError(ValueError):
    """A row exceeds a documented Milvus VARCHAR, array, or vector limit."""


@dataclass(frozen=True)
class _PhysicalField:
    """One Milvus column exactly as EverOS declares it.

    The same descriptor drives collection creation and startup verification,
    so a reported mismatch always means the server disagrees with us — never
    that the builder and the checker have drifted apart from each other.
    """

    name: str
    datatype: DataType
    is_primary: bool = False
    nullable: bool = False
    dim: int | None = None
    element_type: DataType | None = None
    max_length: int | None = None
    max_capacity: int | None = None
    enable_analyzer: bool = False

    def create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"field_name": self.name, "datatype": self.datatype}
        if self.is_primary:
            kwargs["is_primary"] = True
        if self.element_type is not None:
            kwargs["element_type"] = self.element_type
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.max_capacity is not None:
            kwargs["max_capacity"] = self.max_capacity
        if self.dim is not None:
            kwargs["dim"] = self.dim
        if self.enable_analyzer:
            kwargs["enable_analyzer"] = True
        if self.nullable:
            kwargs["nullable"] = True
        return kwargs

    def mismatches(self, actual: dict[str, Any]) -> list[str]:
        """Report how a server-reported field differs from this declaration.

        ``describe_collection`` omits ``nullable`` and ``is_primary`` when they
        are false and omits ``element_type`` for non-array columns, so absence
        is unambiguous and can be compared strictly. ``max_length`` and
        ``max_capacity`` are reported only for the types that carry them and
        the server may normalize them, so they are advisory: a wrong length
        surfaces as a loud write rejection anyway, whereas a wrong datatype or
        dimension is the kind that fails opaquely much later.
        """
        params = actual.get("params") or {}
        out: list[str] = []
        if actual.get("type") != self.datatype:
            out.append(
                f"{self.name}: datatype {_name_of(actual.get('type'))} "
                f"!= expected {_name_of(self.datatype)}"
            )
        if bool(actual.get("is_primary", False)) != self.is_primary:
            out.append(
                f"{self.name}: is_primary {actual.get('is_primary', False)} "
                f"!= expected {self.is_primary}"
            )
        if bool(actual.get("nullable", False)) != self.nullable:
            out.append(
                f"{self.name}: nullable {actual.get('nullable', False)} "
                f"!= expected {self.nullable}"
            )
        if self.dim is not None and params.get("dim") != self.dim:
            out.append(f"{self.name}: dim {params.get('dim')} != expected {self.dim}")
        if self.element_type is not None and actual.get("element_type") != (
            self.element_type
        ):
            out.append(
                f"{self.name}: element_type {_name_of(actual.get('element_type'))} "
                f"!= expected {_name_of(self.element_type)}"
            )
        return out

    def soft_mismatches(self, actual: dict[str, Any]) -> list[str]:
        """Advisory-only limit drift (see :meth:`mismatches`)."""
        params = actual.get("params") or {}
        out: list[str] = []
        for key, want in (
            ("max_length", self.max_length),
            ("max_capacity", self.max_capacity),
        ):
            got = params.get(key)
            if want is not None and got is not None and int(got) != want:
                out.append(f"{self.name}: {key} {got} != declared {want}")
        return out


def _name_of(datatype: Any) -> str:
    return getattr(datatype, "name", repr(datatype))


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
        # Only the readiness set is cleared. Dropping _collection_locks would
        # hand a fresh Lock to the next caller while another task still holds
        # the old one, so ensure_collection would stop being mutually
        # exclusive exactly when a drop/rebuild is in flight.
        cls._ready_collections.clear()

    @classmethod
    def _reset_locks_for_tests(cls) -> None:
        cls._write_locks.clear()
        cls._collection_locks.clear()
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
        """Reject a collection whose physical schema disagrees with ours.

        A name-only check waves through a collection whose columns happen to
        share our names but carry the wrong datatype, primary key, nullability
        or vector dimension — a stale model, or a ``collection_prefix``
        collision with someone else's data. That collection starts fine and
        then fails opaquely on the first write or search. Comparing the
        physical shape turns it back into a startup error with a recovery
        path.
        """
        client = await get_client()
        description = await _run(client.describe_collection, self.collection_name)
        reported = {
            field["name"]: field
            for field in description.get("fields", [])
            if "name" in field
        }
        expected = {physical.name: physical for physical in self._physical_fields()}

        missing = sorted(set(expected) - set(reported))
        stale = sorted(set(reported) - set(expected))
        drift: list[str] = []
        advisory: list[str] = []
        for name, physical in expected.items():
            actual = reported.get(name)
            if actual is None:
                continue
            drift.extend(physical.mismatches(actual))
            advisory.extend(physical.soft_mismatches(actual))

        if advisory:
            logger.warning(
                "milvus_collection_limit_drift",
                collection=self.collection_name,
                details=advisory,
            )
        if missing or stale or drift:
            raise MilvusSchemaMismatchError(
                f"Milvus collection {self.collection_name!r} schema drift: "
                f"missing={missing}, stale={stale}, incompatible={drift}. "
                "The index is rebuildable from markdown; run "
                "`everos cascade rebuild`."
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
        """Read-modify-write the matching rows.

        Milvus has no partial-column update, so the whole row is read back and
        re-upserted. Both halves must hold the same lock: with the read
        outside it, two concurrent updates to one row (backfill writing
        ``vector`` while reflection writes ``deprecated_by``) each overwrite
        the other's column with the value they read before it landed.
        """
        client = await get_client()
        async with self._write_lock(self.collection_name):
            rows = await self._scan_raw(where, include_vectors=True)
            if not rows:
                return
            patched: list[dict[str, Any]] = []
            for row in rows:
                merged = dict(row)
                for key, value in updates.items():
                    self._write_field_value(merged, key, value)
                self._validate_raw_record(merged)
                patched.append(merged)
            await _run(client.upsert, self.collection_name, patched)

    async def optimize(self, *, cleanup_older_than: dt.timedelta | None = None) -> None:
        """Milvus indexes and compaction are service-managed."""

    async def prune(self, older_than: dt.timedelta) -> None:
        """Milvus compaction and retention are service-managed."""

    async def rebuild_indexes(self) -> None:
        """Milvus AUTOINDEX maintenance is service-managed."""

    async def count(self) -> int:
        return await self._count_where(None)

    async def count_where(self, where: Predicate | None = None) -> int:
        return await self._count_where(where)

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

    async def scan(self, where: Predicate | None = None) -> list[T]:
        """Stream every matching row without a fixed query-window cap."""
        rows = await self._scan_raw(where, include_vectors=True)
        return [self._model_from_milvus(row) for row in rows]

    async def _scan_raw(
        self,
        where: Predicate | None,
        *,
        include_vectors: bool,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read through Milvus' iterator, optionally stopping at a soft cap.

        A normal ``query(limit=...)`` cannot cross Milvus' 16,384-row result
        window. The public pagination contract deliberately allows a 20,000
        candidate window, so both maintenance scans and paginated reads must
        use the iterator API.
        """
        await self.ensure_collection()
        client = await get_client()
        iterator = await _run(
            client.query_iterator,
            self.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self._expr(where),
            output_fields=self._output_fields(include_vectors=include_vectors),
        )
        rows: list[dict[str, Any]] = []
        try:
            while batch := await _run(iterator.next):
                rows.extend(batch)
                if max_rows is not None and len(rows) >= max_rows:
                    del rows[max_rows:]
                    break
        finally:
            await _run(iterator.close)
        return rows

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
        raw = await self._scan_raw(
            where,
            include_vectors=True,
            max_rows=max_fetch,
        )
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

    def _build_collection_schema(self) -> Any:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for physical in self._physical_fields():
            schema.add_field(**physical.create_kwargs())
        for field in self.index_schema.bm25_fields:
            sparse_name = _sparse_field(field)
            schema.add_function(
                Function(
                    name=f"{field}_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=[field],
                    output_field_names=[sparse_name],
                )
            )
        return schema

    def _physical_fields(self) -> tuple[_PhysicalField, ...]:
        """Every Milvus column this table declares, in creation order."""
        out: list[_PhysicalField] = []
        for field in self.index_schema.fields:
            out.extend(self._physical_for(field))
        if not self.index_schema.vector_fields:
            out.append(
                _PhysicalField(
                    name=_DUMMY_VECTOR_FIELD,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=_DUMMY_VECTOR_DIMENSION,
                )
            )
        out.extend(
            _PhysicalField(
                name=_sparse_field(field),
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            for field in self.index_schema.bm25_fields
        )
        return tuple(out)

    def _physical_for(self, field: IndexField) -> list[_PhysicalField]:
        is_bm25 = field.name in self.index_schema.bm25_fields
        if field.kind is IndexFieldKind.STRING:
            return [
                _PhysicalField(
                    name=field.name,
                    datatype=DataType.VARCHAR,
                    is_primary=field.primary,
                    # A BM25 input is written as "" rather than null so the
                    # analyzer always has something to tokenize.
                    nullable=field.nullable and not field.primary and not is_bm25,
                    max_length=field.max_length,
                    enable_analyzer=is_bm25,
                )
            ]
        if field.kind is IndexFieldKind.STRING_ARRAY:
            return [
                _PhysicalField(
                    name=field.name,
                    datatype=DataType.ARRAY,
                    nullable=field.nullable,
                    element_type=DataType.VARCHAR,
                    max_length=field.max_length,
                    max_capacity=field.max_capacity,
                )
            ]
        if field.kind is IndexFieldKind.FLOAT:
            return [
                _PhysicalField(
                    name=field.name,
                    datatype=DataType.DOUBLE,
                    nullable=field.nullable,
                )
            ]
        if field.kind is IndexFieldKind.INTEGER:
            return [
                _PhysicalField(
                    name=field.name,
                    datatype=DataType.INT64,
                    nullable=field.nullable,
                )
            ]
        if field.kind is IndexFieldKind.DATETIME:
            return [
                _PhysicalField(
                    name=_datetime_storage_field(field.name),
                    datatype=DataType.INT64,
                    nullable=field.nullable,
                )
            ]
        if field.kind is IndexFieldKind.DENSE_VECTOR:
            return [
                _PhysicalField(
                    name=field.name,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=field.dimension,
                ),
                # Milvus cannot store a null dense vector, so a logical null is
                # a zero vector plus this presence marker.
                _PhysicalField(
                    name=_present_field(field.name),
                    datatype=DataType.BOOL,
                ),
            ]
        # pragma: no cover - enum exhaustiveness guard
        raise TypeError(f"unsupported index field kind: {field.kind}")

    def _stored_field_names(self) -> list[str]:
        return [physical.name for physical in self._physical_fields()]

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
                out[field.name] = (
                    None if value is None else from_timestamp_ms(int(value))
                )
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
            vector_fields={field.name for field in self.index_schema.vector_fields},
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
