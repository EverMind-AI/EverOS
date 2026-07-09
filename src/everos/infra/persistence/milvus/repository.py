"""Milvus repository implementation for EverOS derived index tables."""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar, get_args, get_origin

from pymilvus import DataType, Function, FunctionType, MilvusClient

from everos.component.utils.datetime import (
    ensure_utc,
    from_iso_format,
    from_timestamp,
    to_timestamp_ms,
)
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.persistence import BaseLanceTable

from .milvus_manager import (
    MilvusSchemaMismatchError,
    collection_name,
    get_client,
)

logger = get_logger(__name__)

_DUMMY_VECTOR_FIELD = "_everos_dummy_vector"
_SPARSE_SUFFIX = "__sparse"
_MAX_VARCHAR_LENGTH = 65_535
_ID_MAX_LENGTH = 512
_ARRAY_MAX_CAPACITY = 256


def _q(value: str) -> str:
    """Escape a string for a Milvus single-quoted expression literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


class MilvusRepoBase[T: BaseLanceTable]:
    """Generic Milvus repository for one EverOS derived index collection."""

    schema: type[T]
    _write_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    @property
    def table_name(self) -> str:
        return self.schema.TABLE_NAME

    @property
    def collection_name(self) -> str:
        return collection_name(self.table_name)

    @classmethod
    def _write_lock(cls, name: str) -> asyncio.Lock:
        return cls._write_locks.setdefault(name, asyncio.Lock())

    @classmethod
    def _reset_locks_for_tests(cls) -> None:
        cls._write_locks.clear()

    async def ensure_collection(self) -> None:
        client = await get_client()
        if await _run(client.has_collection, self.collection_name):
            await self.verify_collection()
            return

        schema = self._build_collection_schema()
        index_params = client.prepare_index_params()
        if "vector" in self.schema.model_fields:
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
        else:
            index_params.add_index(
                field_name=_DUMMY_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
        for field in self.schema.BM25_FIELDS:
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
        if missing:
            raise MilvusSchemaMismatchError(
                f"Milvus collection {self.collection_name!r} schema drift: "
                f"missing={sorted(missing)}. The index is rebuildable from md; "
                "drop the collection and restart to rebuild it."
            )

    # ── Create / update ─────────────────────────────────────────────

    async def add(self, records: Sequence[T]) -> None:
        if not records:
            return
        await self.ensure_collection()
        client = await get_client()
        payload = [self._to_milvus_record(record) for record in records]
        async with self._write_lock(self.collection_name):
            await _run(client.insert, self.collection_name, payload)

    async def upsert(self, records: Sequence[T], *, by: str = "id") -> None:
        if by != "id":
            raise ValueError("MilvusRepoBase only supports upsert by id")
        if not records:
            return
        await self.ensure_collection()
        client = await get_client()
        payload = [self._to_milvus_record(record) for record in records]
        async with self._write_lock(self.collection_name):
            await _run(client.upsert, self.collection_name, payload)

    async def update(self, updates: dict[str, Any], *, where: Any) -> None:
        await self.ensure_collection()
        rows = await self._query_raw(where, limit=10_000, include_vectors=True)
        if not rows:
            return
        patched = []
        for row in rows:
            merged = dict(row)
            for key, value in updates.items():
                self._write_field_value(merged, key, value)
            patched.append(merged)
        client = await get_client()
        async with self._write_lock(self.collection_name):
            await _run(client.upsert, self.collection_name, patched)

    # ── Maintenance ────────────────────────────────────────────────

    async def optimize(self, *, cleanup_older_than: dt.timedelta | None = None) -> None:
        """Milvus indexes are managed by the service; no per-write compaction."""

    async def rebuild_indexes(self) -> None:
        """Milvus AUTOINDEX maintenance is service-managed."""

    # ── Reads ──────────────────────────────────────────────────────

    async def count(self) -> int:
        await self.ensure_collection()
        client = await get_client()
        stats = await _run(client.get_collection_stats, self.collection_name)
        row_count = stats.get("row_count", 0)
        return int(row_count)

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> T | None:
        if id_field != "id":
            rows = await self.find_where(f"{id_field} = '{_q(id_value)}'", limit=1)
            return rows[0] if rows else None
        await self.ensure_collection()
        client = await get_client()
        rows = await _run(
            client.get,
            self.collection_name,
            ids=[id_value],
            output_fields=self._output_fields(include_vectors=True),
        )
        if not rows:
            return None
        return self._model_from_milvus(rows[0])

    async def find_where(self, where: Any, *, limit: int = 100) -> list[T]:
        rows = await self._query_raw(where, limit=limit, include_vectors=True)
        return [self._model_from_milvus(row) for row in rows]

    async def find_one_where(self, where: Any) -> T | None:
        rows = await self.find_where(where, limit=1)
        return rows[0] if rows else None

    async def find_where_paginated(
        self,
        where: Any,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20_000,
    ) -> tuple[list[T], int]:
        raw = await self._query_raw(where, limit=max_fetch, include_vectors=True)
        total = len(raw)
        if total >= max_fetch:
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
        return await self.find_where(f"owner_id = '{_q(owner_id)}'", limit=limit)

    async def find_by_md_path(self, md_path: str) -> T | None:
        return await self.find_one_where(f"md_path = '{_q(md_path)}'")

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: Any = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if vector is None:
            return await self._query_candidate_rows(where, limit=limit)
        return await self.dense_search(vector, where, limit=limit)

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Any,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not query_terms:
            return []
        await self.ensure_collection()
        fields = list(columns or self.schema.BM25_FIELDS)
        if not fields:
            return []
        client = await get_client()
        expr = self._expr(where)
        query = " ".join(term for term in query_terms if term)
        best: dict[str, dict[str, Any]] = {}
        for field in fields:
            results = await _run(
                client.search,
                self.collection_name,
                data=[query],
                anns_field=_sparse_field(field),
                filter=expr,
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
        where: Any,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not vector or "vector" not in self.schema.model_fields:
            return []
        await self.ensure_collection()
        client = await get_client()
        results = await _run(
            client.search,
            self.collection_name,
            data=[list(vector)],
            anns_field="vector",
            filter=self._expr(where),
            limit=limit,
            output_fields=self._output_fields(include_vectors=False),
            search_params={"metric_type": "COSINE"},
        )
        return [
            self._candidate_row_from_search(row, normalize_cosine=True)
            for row in _first_result_set(results)
        ]

    # ── Deletes ────────────────────────────────────────────────────

    async def delete(self, predicate: Any) -> None:
        await self.ensure_collection()
        client = await get_client()
        async with self._write_lock(self.collection_name):
            await _run(
                client.delete,
                self.collection_name,
                filter=self._expr(predicate),
            )

    async def delete_by_md_path(self, md_path: str) -> int:
        await self.ensure_collection()
        before = await self.find_where(f"md_path = '{_q(md_path)}'", limit=10_000)
        if not before:
            return 0
        await self.delete(f"md_path = '{_q(md_path)}'")
        return len(before)

    # ── Daily-log helpers ──────────────────────────────────────────

    async def find_by_owner_entry(
        self,
        owner_id: str,
        entry_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> T | None:
        return await self.find_one_where(
            f"owner_id = '{_q(owner_id)}' AND entry_id = '{_q(entry_id)}' "
            f"AND app_id = '{_q(app_id)}' AND project_id = '{_q(project_id)}'"
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
        quoted = ", ".join(f"'{_q(entry_id)}'" for entry_id in entry_ids)
        return await self.find_where(
            f"owner_id = '{_q(owner_id)}' AND entry_id IN ({quoted}) "
            f"AND app_id = '{_q(app_id)}' AND project_id = '{_q(project_id)}'",
            limit=len(entry_ids),
        )

    async def find_by_session(
        self, owner_id: str, session_id: str, *, limit: int = 100
    ) -> list[T]:
        return await self.find_where(
            f"owner_id = '{_q(owner_id)}' AND session_id = '{_q(session_id)}'",
            limit=limit,
        )

    async def find_by_parent(
        self, parent_type: str, parent_id: str, *, limit: int = 100
    ) -> list[T]:
        return await self.find_where(
            f"parent_type = '{_q(parent_type)}' AND parent_id = '{_q(parent_id)}'",
            limit=limit,
        )

    # ── Field conversion ───────────────────────────────────────────

    def _build_collection_schema(self):  # type: ignore[no-untyped-def]
        settings = load_settings().milvus
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        has_vector = False
        for name, field in self.schema.model_fields.items():
            if name == "subject_vector":
                continue
            annotation = field.annotation
            if name == "id":
                schema.add_field(
                    field_name=name,
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=_ID_MAX_LENGTH,
                )
            elif name == "vector":
                has_vector = True
                schema.add_field(
                    field_name=name,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=settings.dimension,
                )
            elif _is_datetime(annotation):
                schema.add_field(
                    field_name=_datetime_storage_field(name),
                    datatype=DataType.INT64,
                    nullable=_is_optional(annotation),
                )
            elif _is_list(annotation):
                schema.add_field(
                    field_name=name,
                    datatype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_capacity=_ARRAY_MAX_CAPACITY,
                    max_length=_ID_MAX_LENGTH,
                    nullable=_is_optional(annotation),
                )
            elif _is_float(annotation):
                schema.add_field(
                    field_name=name,
                    datatype=DataType.DOUBLE,
                    nullable=_is_optional(annotation),
                )
            elif _is_int(annotation):
                schema.add_field(
                    field_name=name,
                    datatype=DataType.INT64,
                    nullable=_is_optional(annotation),
                )
            else:
                kwargs: dict[str, Any] = {
                    "field_name": name,
                    "datatype": DataType.VARCHAR,
                    "max_length": _MAX_VARCHAR_LENGTH,
                }
                if name in self.schema.BM25_FIELDS:
                    kwargs["enable_analyzer"] = True
                elif _is_optional(annotation):
                    kwargs["nullable"] = True
                schema.add_field(**kwargs)

        if not has_vector:
            schema.add_field(
                field_name=_DUMMY_VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=1,
            )
        for field in self.schema.BM25_FIELDS:
            schema.add_field(
                field_name=_sparse_field(field),
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_function(
                Function(
                    name=f"{field}_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=[field],
                    output_field_names=[_sparse_field(field)],
                )
            )
        return schema

    def _stored_field_names(self) -> list[str]:
        names: list[str] = []
        has_vector = False
        for name, field in self.schema.model_fields.items():
            if name == "subject_vector":
                continue
            if name == "vector":
                has_vector = True
                names.append(name)
            elif _is_datetime(field.annotation):
                names.append(_datetime_storage_field(name))
            else:
                names.append(name)
        if not has_vector:
            names.append(_DUMMY_VECTOR_FIELD)
        names.extend(_sparse_field(field) for field in self.schema.BM25_FIELDS)
        return names

    def _output_fields(self, *, include_vectors: bool) -> list[str]:
        fields: list[str] = []
        for name in self._stored_field_names():
            if name.endswith(_SPARSE_SUFFIX) or name == _DUMMY_VECTOR_FIELD:
                continue
            if not include_vectors and name == "vector":
                continue
            fields.append(name)
        return fields

    def _to_milvus_record(self, record: T) -> dict[str, Any]:
        raw = record.model_dump(mode="python")
        out: dict[str, Any] = {}
        for name, field in self.schema.model_fields.items():
            if name == "subject_vector":
                continue
            value = raw.get(name)
            if name in self.schema.BM25_FIELDS and value is None:
                value = ""
            if _is_datetime(field.annotation):
                storage_name = _datetime_storage_field(name)
                out[storage_name] = (
                    _datetime_to_ms(value) if value is not None else None
                )
            elif name == "vector":
                out[name] = list(value or [])
            elif _is_list(field.annotation):
                out[name] = [str(item) for item in (value or [])]
            else:
                out[name] = value
        if "vector" not in self.schema.model_fields:
            out[_DUMMY_VECTOR_FIELD] = [0.0]
        return out

    def _model_from_milvus(self, row: dict[str, Any]) -> T:
        shaped = self._restore_row(row, include_distance=False)
        return self.schema.model_validate(shaped)

    def _candidate_row_from_search(
        self, row: dict[str, Any], *, normalize_cosine: bool = False
    ) -> dict[str, Any]:
        shaped = self._restore_row(row.get("entity", {}), include_distance=False)
        raw_distance = row.get("distance")
        shaped["_distance"] = (
            _cosine_distance_from_milvus(raw_distance)
            if normalize_cosine
            else raw_distance
        )
        return shaped

    def _restore_row(
        self, row: dict[str, Any], *, include_distance: bool
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, field in self.schema.model_fields.items():
            if name == "subject_vector":
                continue
            if _is_datetime(field.annotation):
                storage_name = _datetime_storage_field(name)
                value = row.get(storage_name)
                if value is None:
                    out[name] = None
                else:
                    out[name] = from_timestamp(int(value))
            elif name in row:
                value = row[name]
                if name in self.schema.BM25_FIELDS and value == "" and _is_optional(
                    field.annotation
                ):
                    out[name] = None
                else:
                    out[name] = value
        if include_distance and "distance" in row:
            out["_distance"] = row["distance"]
        return out

    def _write_field_value(
        self, row: dict[str, Any], field_name: str, value: Any
    ) -> None:
        field = self.schema.model_fields.get(field_name)
        if field is None:
            row[field_name] = value
        elif _is_datetime(field.annotation):
            row[_datetime_storage_field(field_name)] = (
                _datetime_to_ms(value) if value is not None else None
            )
        else:
            row[field_name] = value

    async def _query_raw(
        self, where: Any, *, limit: int, include_vectors: bool
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
        self, where: Any, *, limit: int
    ) -> list[dict[str, Any]]:
        rows = await self._query_raw(where, limit=limit, include_vectors=False)
        return [self._restore_row(row, include_distance=False) for row in rows]

    def _expr(self, where: Any) -> str:
        if where is None:
            return ""
        milvus_expr = getattr(where, "milvus", None)
        if isinstance(milvus_expr, str):
            return milvus_expr
        if not isinstance(where, str):
            return str(where)
        return _lancedb_expr_to_milvus(where)


def _lancedb_expr_to_milvus(expr: str) -> str:
    """Best-effort conversion for internal EverOS LanceDB predicates."""
    converted = expr
    converted = re.sub(r"\barray_has\s*\(", "array_contains(", converted)
    converted = re.sub(r"\bIS\s+NULL\b", "is null", converted, flags=re.IGNORECASE)
    converted = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IN\s+\(([^)]*)\)",
        lambda m: f"{m.group(1)} in [{m.group(2)}]",
        converted,
        flags=re.IGNORECASE,
    )
    converted = re.sub(r"(?<![!<>=])=(?!=)", "==", converted)
    converted = re.sub(r"\btimestamp\b", "timestamp_ms", converted)
    converted = re.sub(r"\bcreated_at\b", "created_at_ms", converted)
    converted = re.sub(r"\bupdated_at\b", "updated_at_ms", converted)
    converted = re.sub(
        r"TIMESTAMP\s+'([^']*)'",
        lambda m: str(_iso_to_ms(m.group(1))),
        converted,
    )
    return converted


def _datetime_storage_field(name: str) -> str:
    return f"{name}_ms"


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


def _iso_to_ms(value: str) -> int:
    aware = ensure_utc(from_iso_format(value))
    assert aware is not None
    return to_timestamp_ms(aware)


def _is_optional(annotation: Any) -> bool:
    args = get_args(annotation)
    return type(None) in args


def _non_none_args(annotation: Any) -> tuple[Any, ...]:
    return tuple(arg for arg in get_args(annotation) if arg is not type(None))


def _is_datetime(annotation: Any) -> bool:
    if annotation is dt.datetime:
        return True
    return any(arg is dt.datetime for arg in _non_none_args(annotation))


def _is_list(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is list:
        return True
    return any(get_origin(arg) is list for arg in _non_none_args(annotation))


def _is_float(annotation: Any) -> bool:
    if annotation is float:
        return True
    return any(arg is float for arg in _non_none_args(annotation))


def _is_int(annotation: Any) -> bool:
    if annotation is int:
        return True
    return any(arg is int for arg in _non_none_args(annotation))


def _sort_value(value: Any) -> Any:
    if value is None:
        fallback = ensure_utc(dt.datetime.min)
        assert fallback is not None
        return fallback
    return value


def _bm25_score_from_distance(distance: Any) -> float:
    """Convert the Milvus BM25 search distance into a higher-is-better score."""
    if distance is None:
        return 0.0
    return abs(float(distance))


def _cosine_distance_from_milvus(distance: Any) -> float | None:
    """Normalize Milvus COSINE results to LanceDB-style distance.

    Milvus server / Zilliz Cloud return cosine similarity through the
    ``distance`` field. Milvus Lite 3.0 reports cosine distance instead; keep
    this version-gated workaround narrow because the upstream issue is expected
    to be fixed after 3.0:
    https://github.com/milvus-io/milvus-lite/issues/343

    Recaller code expects LanceDB's ``1 - similarity`` distance contract.
    """
    if distance is None:
        return None
    value = float(distance)
    if not _uses_milvus_lite_3_0_cosine_distance_bug():
        value = 1.0 - value
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _uses_milvus_lite() -> bool:
    uri = load_settings().milvus.uri
    if not uri:
        return True
    return "://" not in uri


def _uses_milvus_lite_3_0_cosine_distance_bug() -> bool:
    if not _uses_milvus_lite():
        return False
    return _milvus_lite_version() in {"3.0", "3.0.0"}


def _milvus_lite_version() -> str | None:
    try:
        return version("milvus-lite")
    except PackageNotFoundError:
        return None


def _first_result_set(results: Any) -> list[dict[str, Any]]:
    if not results:
        return []
    first = results[0]
    return list(first or [])


async def _run(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)


__all__ = ["MilvusRepoBase"]
