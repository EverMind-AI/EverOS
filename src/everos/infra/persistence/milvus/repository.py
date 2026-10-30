"""Milvus repository for EverOS rebuildable derived indexes.

:class:`MilvusRepoBase` is the async repository surface one table's index
repository exposes (CRUD, scans, pagination, dense / BM25 search) over a
Milvus collection. It owns the IO — the client calls, the per-collection
write and readiness locks — and delegates the pure halves to two
collaborators built from the table's index schema: a
:class:`~._layout.CollectionLayout` that declares and verifies the collection,
and a :class:`~._codec.RowCodec` that translates records to rows and back.
"""

from __future__ import annotations

__all__ = ["MilvusRepoBase", "MilvusValueLimitError"]

import asyncio
import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar, overload

from pydantic import BaseModel
from pymilvus import MilvusClient

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import ensure_utc
from everos.infra.persistence.index.schema import (
    IndexFieldKind,
    IndexSchema,
    schema_for,
)
from everos.memory.ports.query import (
    Filter,
    all_of,
    eq,
    one_of,
)

from ._codec import (
    MilvusValueLimitError,
    RowCodec,
    bm25_score_from_distance,
    compose_pk,
)
from ._columns import PK_FIELD, present_field, sparse_field
from ._layout import CollectionLayout
from .milvus_manager import (
    MilvusConnection,
    collection_name,
)
from .predicate import render_predicate

logger = get_logger(__name__)

_SEARCH_TOPK_MAX = 16_384
_SEARCH_TOPK_MAX = 16_384
"""Milvus caps ``search`` topK at its result window and rejects anything
larger outright. LanceDB has no such ceiling, so callers pass generous
pool sizes -- agentic search deliberately passes a large sentinel and
expects the engine to clamp. Clamping here keeps that expectation true on
both backends instead of making every caller learn one engine's limit."""


class MilvusRepoBase[T: BaseModel]:
    """Generic Milvus repository backed by one neutral index schema.

    Args:
        connection: The application's Milvus connection.
        dim: ``[embedding] dim`` -- the width of the dense vector fields.
    """

    schema: type[T]
    _write_locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _collection_locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _ready_collections: ClassVar[set[str]] = set()

    @property
    def index_schema(self) -> IndexSchema:
        return self._index_schema

    @property
    def table_name(self) -> str:
        return self.index_schema.table_name

    def __init__(self, connection: MilvusConnection, *, dim: int) -> None:
        self._connection = connection
        self._index_schema = schema_for(self.schema, dim=dim)
        self._layout = CollectionLayout(self.index_schema)
        self._codec = RowCodec(self.schema, self.index_schema)

    @property
    def collection_name(self) -> str:
        return collection_name(self.table_name, self._connection.settings)

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
            client = await self._connection.client()
            if await _run(client.has_collection, name):
                await self.verify_collection()
            else:
                await self._create_collection(client)
            self._ready_collections.add(name)

    async def _create_collection(self, client: MilvusClient) -> None:
        index_params = client.prepare_index_params()
        self._layout.add_indexes(index_params)
        settings = self._connection.settings
        await _run(
            client.create_collection,
            collection_name=self.collection_name,
            schema=self._layout.build_schema(),
            index_params=index_params,
            consistency_level=settings.consistency_level,
        )
        logger.info(
            "milvus_collection_created",
            table=self.table_name,
            collection=self.collection_name,
        )

    async def verify_collection(self) -> None:
        """Reject a collection whose physical metadata disagrees with ours.

        Fetches the server's description of the collection and of each of its
        indexes and checks both against :class:`~._layout.CollectionLayout`;
        see :meth:`~._layout.CollectionLayout.verify_fields` for why a
        name-only check is not enough.

        Raises:
            MilvusSchemaMismatchError: The collection or one of its indexes
                differs from the declaration.
        """
        client = await self._connection.client()
        name = self.collection_name
        description = await _run(client.describe_collection, name)
        self._layout.verify_fields(description, collection_name=name)
        index_names = await _run(client.list_indexes, name)
        index_descriptions = [
            (index_name, await _run(client.describe_index, name, index_name))
            for index_name in index_names
        ]
        self._layout.verify_indexes(index_descriptions, collection_name=name)

    async def upsert(self, records: Sequence[T]) -> None:
        if not records:
            return
        await self.ensure_collection()
        payload = [self._codec.to_record(record) for record in records]
        client = await self._connection.client()
        async with self._write_lock(self.collection_name):
            await _run(client.upsert, self.collection_name, payload)

    async def update(self, updates: dict[str, Any], *, where: Filter) -> None:
        """Read-modify-write the matching rows.

        Milvus has no partial-column update, so the whole row is read back and
        re-upserted. Both halves must hold the same lock: with the read
        outside it, two concurrent updates to one row (backfill writing
        ``vector`` while reflection writes ``deprecated_by``) each overwrite
        the other's column with the value they read before it landed.

        A primary-key column in ``updates`` is refused: the re-upsert would
        land under a new ``__pk`` and leave the old row behind.
        """
        clash = sorted(set(updates) & set(self.index_schema.primary_key))
        if clash:
            raise ValueError(
                f"update() cannot change primary-key column(s) {clash} of "
                f"{self.collection_name}; delete the row and upsert it instead"
            )
        client = await self._connection.client()
        async with self._write_lock(self.collection_name):
            rows = await self._scan_raw(where, include_vectors=True)
            if not rows:
                return
            patched: list[dict[str, Any]] = []
            for row in rows:
                merged = dict(row)
                for key, value in updates.items():
                    self._codec.write_field_value(merged, key, value)
                merged[PK_FIELD] = compose_pk(self.index_schema, merged)
                self._codec.validate_raw(merged)
                patched.append(merged)
            await _run(client.upsert, self.collection_name, patched)

    async def optimize(self, *, cleanup_older_than: dt.timedelta | None = None) -> None:
        """Milvus indexes and compaction are service-managed."""

    async def prune(self, older_than: dt.timedelta) -> None:
        """Milvus compaction and retention are service-managed."""

    async def rebuild_indexes(self) -> None:
        """Milvus AUTOINDEX maintenance is service-managed."""

    async def ensure_vector_indexes(self) -> None:
        """Milvus builds its vector index at collection creation."""

    async def count(self) -> int:
        return await self._count_where(None)

    async def count_where(self, where: Filter | None = None) -> int:
        return await self._count_where(where)

    async def _count_where(self, where: Filter | None) -> int:
        await self.ensure_collection()
        client = await self._connection.client()
        rows = await _run(
            client.query,
            self.collection_name,
            filter=self._expr(where),
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    async def find_where(
        self, where: Filter, *, limit: int = 100, skip_vectors: bool = False
    ) -> list[T]:
        rows = await self._query_raw(
            where, limit=limit, include_vectors=not skip_vectors
        )
        return [self._codec.to_model(row) for row in rows]

    async def find_one_where(self, where: Filter) -> T | None:
        rows = await self.find_where(where, limit=1)
        return rows[0] if rows else None

    async def scan(self, where: Filter | None = None) -> list[T]:
        """Stream every matching row without a fixed query-window cap."""
        rows = await self._scan_raw(where, include_vectors=True)
        return [self._codec.to_model(row) for row in rows]

    async def _scan_raw(
        self,
        where: Filter | None,
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
        client = await self._connection.client()
        iterator = await _run(
            client.query_iterator,
            self.collection_name,
            batch_size=1000,
            limit=-1,
            filter=self._expr(where),
            output_fields=self._layout.output_fields(include_vectors=include_vectors),
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
        where: Filter,
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
                "find_where_paginated truncated",
                table=self.table_name,
                total=total,
                max_fetch=max_fetch,
            )
        rows = [self._codec.to_model(row) for row in raw]
        rows.sort(
            key=lambda row: _sort_value(getattr(row, sort_by, None)),
            reverse=descending,
        )
        offset = (page - 1) * page_size
        return rows[offset : offset + page_size], total

    async def find_by_md_path(self, md_path: str) -> T | None:
        return await self.find_one_where(eq("md_path", md_path))

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: Filter | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if vector is None:
            return await self._query_candidate_rows(where, limit=limit)
        return await self.dense_search(vector, where, limit=limit)

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Filter | None,
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
        client = await self._connection.client()
        query = " ".join(term for term in query_terms if term)
        best: dict[str, dict[str, Any]] = {}
        for field in fields:
            results = await _run(
                client.search,
                self.collection_name,
                data=[query],
                anns_field=sparse_field(field),
                filter=self._expr(where),
                limit=min(limit, _SEARCH_TOPK_MAX),
                output_fields=self._layout.output_fields(include_vectors=False),
                search_params={"metric_type": "BM25"},
            )
            for row in _first_result_set(results):
                shaped = self._codec.candidate_from_search(row)
                score = bm25_score_from_distance(row.get("distance"))
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
        where: Filter | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        if not vector:
            return []
        field = self.index_schema.field(vector_field)
        if field.kind is not IndexFieldKind.DENSE_VECTOR:
            raise ValueError(f"{vector_field!r} is not a dense-vector field")
        self._codec.validate_vector(field, vector)
        await self.ensure_collection()
        client = await self._connection.client()
        present = eq(present_field(vector_field), True)
        results = await _run(
            client.search,
            self.collection_name,
            data=[list(vector)],
            anns_field=vector_field,
            filter=self._expr(all_of(where, present)),
            limit=min(limit, _SEARCH_TOPK_MAX),
            output_fields=self._layout.output_fields(include_vectors=False),
            search_params={"metric_type": "COSINE"},
        )
        return [
            self._codec.candidate_from_search(row, normalize_cosine=True)
            for row in _first_result_set(results)
        ]

    async def delete(self, predicate: Filter) -> None:
        await self.ensure_collection()
        client = await self._connection.client()
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

    @overload
    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
        columns: None = None,
        skip_vectors: bool = False,
    ) -> list[T]: ...

    @overload
    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
        columns: Sequence[str],
        skip_vectors: bool = False,
    ) -> list[dict[str, Any]]: ...

    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
        columns: Sequence[str] | None = None,
        skip_vectors: bool = False,
    ) -> list[T] | list[dict[str, Any]]:
        if not entry_ids:
            return []
        if columns is not None and not skip_vectors:
            # A ``columns`` list that names no vector column cannot need the
            # vector payload. Milvus charges for that payload on the fetch,
            # not on the trim below, so decide it here -- otherwise a caller
            # projecting two scalars still materialises every 1024-D vector
            # in the result set and discards it one line later.
            vector_names = {field.name for field in self.index_schema.vector_fields}
            skip_vectors = not (set(columns) & vector_names)
        rows = await self.find_where(
            all_of(
                eq("owner_id", owner_id),
                one_of("entry_id", list(entry_ids)),
                eq("app_id", app_id),
                eq("project_id", project_id),
            ),
            limit=len(entry_ids),
            skip_vectors=skip_vectors,
        )
        if columns is None:
            return rows
        # Milvus has no column projection on the result shape, so the trim to
        # ``columns`` happens in Python for LanceDB return-shape parity; the
        # fetch itself was already narrowed above.
        return [{name: getattr(row, name) for name in columns} for row in rows]

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

    async def _query_raw(
        self,
        where: Filter | None,
        *,
        limit: int,
        include_vectors: bool,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection()
        client = await self._connection.client()
        return await _run(
            client.query,
            self.collection_name,
            filter=self._expr(where),
            output_fields=self._layout.output_fields(include_vectors=include_vectors),
            limit=limit,
        )

    async def _query_candidate_rows(
        self, where: Filter | None, *, limit: int
    ) -> list[dict[str, Any]]:
        rows = await self._query_raw(where, limit=limit, include_vectors=False)
        return [self._codec.restore_row(row) for row in rows]

    def _expr(self, where: Filter | None) -> str:
        if where is not None and not isinstance(where, Filter):
            raise TypeError(
                "Milvus repository predicates must use the neutral Filter AST, "
                f"got {type(where).__name__}"
            )
        return render_predicate(
            where,
            datetime_fields=self.index_schema.datetime_fields,
            vector_fields={field.name for field in self.index_schema.vector_fields},
        )


def _sort_value(value: Any) -> Any:
    if value is None:
        fallback = ensure_utc(dt.datetime.min)
        assert fallback is not None
        return fallback
    return value


def _first_result_set(results: Any) -> list[dict[str, Any]]:
    if not results:
        return []
    return list(results[0] or [])


async def _run(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)
