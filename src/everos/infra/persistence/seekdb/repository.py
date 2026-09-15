"""SQL-first SeekDB repository implementing the complete index port.

Every business table is created from the backend-neutral logical schema. Reads
always name their columns explicitly so tuple rows from embedded pylibseekdb
and dictionary rows from remote PyMySQL have identical behavior.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
    schema_for,
)
from everos.infra.persistence.predicate import Predicate, all_of, eq, one_of

from .codec import from_row, model_from_row, to_row, write_field_value
from .errors import SeekdbSchemaMismatchError
from .predicate import render_predicate
from .schema import (
    build_create_table,
    column_drift,
    index_drift,
    physical_column,
    physical_columns,
    physical_indexes,
)
from .seekdb_manager import get_session, run, table_name
from .sql import literal, quote_identifier

logger = get_logger(__name__)

_SCAN_BATCH_SIZE = 1_000
_SEARCH_LIMIT_MAX = 16_384
_WRITE_BATCH_SIZE = 64


class SeekdbRepoBase[T: BaseModel]:
    """Generic SeekDB repository backed by one portable record model."""

    schema: type[T]
    _ready_tables: ClassVar[set[str]] = set()
    _table_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    @property
    def index_schema(self) -> IndexSchema:
        return schema_for(self.schema)

    @property
    def table_name(self) -> str:
        return self.index_schema.table_name

    @property
    def physical_table_name(self) -> str:
        return table_name(self.table_name)

    @classmethod
    def _table_lock(cls, name: str) -> asyncio.Lock:
        return cls._table_locks.setdefault(name, asyncio.Lock())

    @classmethod
    def _reset_table_cache(cls) -> None:
        cls._ready_tables.clear()

    @classmethod
    def _reset_locks_for_tests(cls) -> None:
        cls._table_locks.clear()
        cls._reset_table_cache()

    async def ensure_table(self) -> None:
        """Create or drift-check this table once per process."""
        name = self.physical_table_name
        if name in self._ready_tables:
            return
        async with self._table_lock(name):
            if name in self._ready_tables:
                return
            if await self.table_exists():
                await self.verify_table()
            else:
                await self._execute(
                    build_create_table(
                        name,
                        self.index_schema,
                        self._vector_sync_mode(),
                    )
                )
                logger.info(
                    "seekdb_table_created", table=self.table_name, physical_table=name
                )
            self._ready_tables.add(name)

    async def table_exists(self) -> bool:
        sql = (
            "SELECT COUNT(*) AS n FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = "
            f"{literal(self.physical_table_name)}"
        )
        return bool(int(await self._fetch_scalar(sql) or 0))

    async def verify_table(self) -> None:
        """Reject column or index drift with a rebuild-oriented error."""
        table = self.physical_table_name
        columns = await self._fetch_all(
            "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, "
            "CHARACTER_SET_NAME, COLLATION_NAME "
            "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() "
            f"AND TABLE_NAME = {literal(table)} ORDER BY ORDINAL_POSITION",
            [
                "COLUMN_NAME",
                "COLUMN_TYPE",
                "IS_NULLABLE",
                "COLUMN_KEY",
                "CHARACTER_SET_NAME",
                "COLLATION_NAME",
            ],
        )
        indexes = await self._fetch_all(
            "SELECT INDEX_NAME, COLUMN_NAME, INDEX_TYPE, SEQ_IN_INDEX, SUB_PART "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() "
            f"AND TABLE_NAME = {literal(table)} ORDER BY INDEX_NAME, SEQ_IN_INDEX",
            ["INDEX_NAME", "COLUMN_NAME", "INDEX_TYPE", "SEQ_IN_INDEX", "SUB_PART"],
        )
        create_rows = await self._fetch_all(
            f"SHOW CREATE TABLE {quote_identifier(table)}", ["Table", "Create Table"]
        )
        create_value = create_rows[0].get("Create Table") if create_rows else ""
        create_sql = (
            create_value.decode("utf-8")
            if isinstance(create_value, bytes)
            else str(create_value or "")
        )
        missing, stale, incompatible = column_drift(
            physical_columns(self.index_schema), columns
        )
        missing_indexes, stale_indexes, incompatible_indexes = index_drift(
            physical_indexes(self.index_schema, self._vector_sync_mode()),
            indexes,
            create_sql=create_sql,
        )
        if (
            missing
            or stale
            or incompatible
            or missing_indexes
            or stale_indexes
            or incompatible_indexes
        ):
            details = (
                f"missing_columns={missing}, stale_columns={stale}, "
                f"incompatible_columns={incompatible}, "
                f"missing_indexes={missing_indexes}, "
                f"stale_indexes={stale_indexes}, "
                f"incompatible_indexes={incompatible_indexes}"
            )
            logger.error("seekdb_schema_drift", table=table, details=details)
            raise SeekdbSchemaMismatchError(
                f"SeekDB table {table!r} schema drift: {details}. The index is "
                "rebuildable from markdown; run `everos cascade rebuild`."
            )

    async def add(self, records: Sequence[T]) -> None:
        if not records:
            return
        await self.ensure_table()
        await self._write_records(records, upsert=False)

    async def upsert(self, records: Sequence[T], *, by: str = "id") -> None:
        if by != "id":
            raise ValueError("SeekdbRepoBase only supports upsert by id")
        if not records:
            return
        await self.ensure_table()
        await self._write_records(records, upsert=True)

    async def _write_records(self, records: Sequence[T], *, upsert: bool) -> None:
        rows = [to_row(record, self.index_schema) for record in records]
        for start in range(0, len(rows), _WRITE_BATCH_SIZE):
            sql = self._insert_sql(rows[start : start + _WRITE_BATCH_SIZE], upsert)
            await self._execute(sql)

    def _insert_sql(self, rows: Sequence[Mapping[str, Any]], upsert: bool) -> str:
        columns = self._stored_columns(include_vectors=True)
        column_sql = ", ".join(quote_identifier(name) for name in columns)
        values = ", ".join(
            "(" + ", ".join(literal(row[name]) for name in columns) + ")"
            for row in rows
        )
        sql = (
            f"INSERT INTO {quote_identifier(self.physical_table_name)} "
            f"({column_sql}) VALUES {values}"
        )
        if upsert:
            primary = next(
                column.name
                for column in physical_columns(self.index_schema)
                if column.primary
            )
            # seekdb and OceanBase's supported MySQL dialect intentionally uses
            # VALUES(col). MySQL upstream deprecates it, but adopting MySQL 8's
            # row-alias syntax would make current seekdb releases incompatible.
            updates = ", ".join(
                f"{quote_identifier(name)}=VALUES({quote_identifier(name)})"
                for name in columns
                if name != primary
            )
            sql += f" ON DUPLICATE KEY UPDATE {updates}"
        return sql

    async def count(self) -> int:
        return await self.count_where()

    async def count_where(self, where: Predicate | None = None) -> int:
        await self.ensure_table()
        sql = f"SELECT COUNT(*) AS n FROM {quote_identifier(self.physical_table_name)}"
        sql += self._where_clause(where)
        return int(await self._fetch_scalar(sql) or 0)

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> T | None:
        self.index_schema.field(id_field)
        return await self.find_one_where(eq(id_field, id_value))

    async def find_where(self, where: Predicate, *, limit: int = 100) -> list[T]:
        rows = await self._query_rows(
            where,
            columns=self._stored_columns(include_vectors=True),
            order_by=f"{quote_identifier('id')} ASC",
            limit=max(0, limit),
        )
        return [self._model(row) for row in rows]

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
        """Page natively; ``max_fetch`` is retained only for port compatibility."""
        _ = max_fetch
        if page < 1 or page_size < 1:
            raise ValueError("page and page_size must be positive")
        field = self.index_schema.field(sort_by)
        column = physical_column(self.index_schema, sort_by)
        if column.dimension is not None or column.sql_type in {"JSON", "LONGTEXT"}:
            raise ValueError(
                f"SeekDB cannot paginate by non-scalar field {sort_by!r} "
                f"({column.sql_type})"
            )
        physical = self._storage_name(field)
        direction = "DESC" if descending else "ASC"
        order = f"{quote_identifier(physical)} {direction}"
        if physical != "id":
            order += f", {quote_identifier('id')} ASC"
        total = await self.count_where(where)
        rows = await self._query_rows(
            where,
            columns=self._stored_columns(include_vectors=True),
            order_by=order,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        return [self._model(row) for row in rows], total

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: Predicate | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if vector is not None:
            return await self.dense_search(vector, where, limit=limit)
        rows = await self._query_rows(
            where,
            columns=self._stored_columns(include_vectors=False),
            order_by=f"{quote_identifier('id')} ASC",
            limit=max(0, limit),
        )
        return [from_row(row, self.index_schema) for row in rows]

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Predicate | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        if not vector or limit <= 0:
            return []
        field = self.index_schema.field(vector_field)
        if field.kind is not IndexFieldKind.DENSE_VECTOR:
            raise ValueError(f"{vector_field!r} is not a dense-vector field")
        _, encoded = write_field_value(field, vector, self.index_schema)
        await self.ensure_table()
        columns = self._stored_columns(include_vectors=False)
        select = self._select_clause(columns)
        vector_column = quote_identifier(vector_field)
        distance = f"cosine_distance({vector_column}, {literal(encoded)})"
        predicate = self._render(where)
        conditions = [
            item for item in (predicate, f"{vector_column} IS NOT NULL") if item
        ]
        sql = (
            f"SELECT {select}, {distance} AS _distance FROM "
            f"{quote_identifier(self.physical_table_name)} WHERE "
            + " AND ".join(f"({item})" for item in conditions)
            + f" ORDER BY _distance APPROXIMATE LIMIT {min(limit, _SEARCH_LIMIT_MAX)}"
        )
        rows = await self._fetch_all(sql, [*columns, "_distance"])
        return [self._search_row(row, score_field="_distance") for row in rows]

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Predicate | None,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        terms = [term for term in query_terms if term]
        fields = list(columns or self.index_schema.bm25_fields)
        if not terms or not fields or limit <= 0:
            return []
        unknown = set(fields) - set(self.index_schema.bm25_fields)
        if unknown:
            raise ValueError(f"unknown BM25 fields: {sorted(unknown)}")
        await self.ensure_table()
        query = literal(" ".join(terms))
        output = self._stored_columns(include_vectors=False)
        select = self._select_clause(output)
        predicate = self._render(where)
        best: dict[str, dict[str, Any]] = {}
        for field in fields:
            match = (
                f"MATCH({quote_identifier(field)}) AGAINST "
                f"({query} IN NATURAL LANGUAGE MODE)"
            )
            conditions = [match, *([predicate] if predicate else [])]
            sql = (
                f"SELECT {select}, {match} AS _score FROM "
                f"{quote_identifier(self.physical_table_name)} WHERE "
                + " AND ".join(f"({item})" for item in conditions)
                + f" ORDER BY _score DESC LIMIT {min(limit, _SEARCH_LIMIT_MAX)}"
            )
            rows = await self._fetch_all(sql, [*output, "_score"])
            for row in rows:
                shaped = self._search_row(row, score_field="_score")
                rid = shaped.get("id")
                if not isinstance(rid, str):
                    continue
                prior = best.get(rid)
                if prior is None or shaped["_score"] > prior["_score"]:
                    best[rid] = shaped
        return sorted(best.values(), key=lambda item: item["_score"], reverse=True)[
            :limit
        ]

    async def scan(self, where: Predicate | None = None) -> list[T]:
        await self.ensure_table()
        columns = self._stored_columns(include_vectors=True)
        predicate = self._render(where)
        rows: list[T] = []
        last_id: str | None = None
        while True:
            conditions = [predicate] if predicate else []
            if last_id is not None:
                conditions.append(f"{quote_identifier('id')} > {literal(last_id)}")
            sql = (
                f"SELECT {self._select_clause(columns)} FROM "
                f"{quote_identifier(self.physical_table_name)}"
            )
            if conditions:
                sql += " WHERE " + " AND ".join(f"({item})" for item in conditions)
            sql += f" ORDER BY {quote_identifier('id')} ASC LIMIT {_SCAN_BATCH_SIZE}"
            batch = await self._fetch_all(sql, columns)
            if not batch:
                break
            models = [self._model(row) for row in batch]
            rows.extend(models)
            last_id = str(models[-1].model_dump(mode="python")["id"])
            if len(batch) < _SCAN_BATCH_SIZE:
                break
        return rows

    async def update(self, updates: dict[str, Any], *, where: Predicate) -> None:
        if not isinstance(where, Predicate):
            raise TypeError("SeekDB update requires a neutral Predicate")
        if not updates:
            return
        await self.ensure_table()
        assignments: list[str] = []
        for name, value in updates.items():
            field = self.index_schema.field(name)
            storage_name, encoded = write_field_value(field, value, self.index_schema)
            assignments.append(f"{quote_identifier(storage_name)} = {literal(encoded)}")
        sql = (
            f"UPDATE {quote_identifier(self.physical_table_name)} SET "
            + ", ".join(assignments)
            + f" WHERE {self._render(where)}"
        )
        await self._execute(sql)

    async def delete(self, predicate: Predicate) -> None:
        if not isinstance(predicate, Predicate):
            raise TypeError("SeekDB delete requires a neutral Predicate")
        await self.ensure_table()
        sql = (
            f"DELETE FROM {quote_identifier(self.physical_table_name)} "
            f"WHERE {self._render(predicate)}"
        )
        await self._execute(sql)

    async def delete_by_md_path(self, md_path: str) -> int:
        predicate = eq("md_path", md_path)
        count = await self.count_where(predicate)
        if count:
            await self.delete(predicate)
        return count

    async def optimize(self) -> None:
        """Immediate vector indexes require no explicit refresh."""

    async def prune(self, older_than: dt.timedelta) -> None:
        """SeekDB owns physical compaction and retention."""
        _ = older_than

    async def rebuild_indexes(self) -> None:
        """SeekDB owns physical index maintenance."""

    async def find_by_owner(self, owner_id: str, *, limit: int = 100) -> list[T]:
        return await self.find_where(eq("owner_id", owner_id), limit=limit)

    async def find_by_md_path(self, md_path: str) -> T | None:
        return await self.find_one_where(eq("md_path", md_path))

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

    async def _query_rows(
        self,
        where: Predicate | None,
        *,
        columns: Sequence[str],
        order_by: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        await self.ensure_table()
        sql = (
            f"SELECT {self._select_clause(columns)} FROM "
            f"{quote_identifier(self.physical_table_name)}" + self._where_clause(where)
        )
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {max(0, limit)}"
        if offset is not None:
            sql += f" OFFSET {max(0, offset)}"
        return await self._fetch_all(sql, columns)

    def _model(self, row: Mapping[str, Any]) -> T:
        return cast(T, model_from_row(row, self.index_schema, self.schema))

    def _search_row(
        self, row: Mapping[str, Any], *, score_field: str
    ) -> dict[str, Any]:
        shaped = from_row(row, self.index_schema)
        raw = row.get(score_field)
        score = 0.0 if raw is None else float(raw)
        shaped[score_field] = max(0.0, score) if score_field == "_score" else score
        return shaped

    def _stored_columns(self, *, include_vectors: bool) -> list[str]:
        return [
            column.name
            for column in physical_columns(self.index_schema)
            if include_vectors or column.dimension is None
        ]

    def _vector_sync_mode(self) -> Literal["immediate", "async"]:
        return load_settings().seekdb.vector_sync_mode

    def _select_clause(self, columns: Sequence[str]) -> str:
        return ", ".join(quote_identifier(name) for name in columns)

    def _where_clause(self, where: Predicate | None) -> str:
        rendered = self._render(where)
        return f" WHERE {rendered}" if rendered else ""

    def _render(self, where: Predicate | None) -> str:
        return render_predicate(
            where,
            datetime_fields=self.index_schema.datetime_fields,
            vector_fields={field.name for field in self.index_schema.vector_fields},
        )

    def _storage_name(self, field: IndexField) -> str:
        return (
            f"{field.name}_ms" if field.kind is IndexFieldKind.DATETIME else field.name
        )

    async def _execute(self, sql: str) -> None:
        session = await get_session()
        await run(session.execute, sql, table=self.physical_table_name)

    async def _fetch_all(
        self, sql: str, columns: Sequence[str]
    ) -> list[dict[str, Any]]:
        session = await get_session()
        return await run(
            session.fetch_all,
            sql,
            columns,
            table=self.physical_table_name,
        )

    async def _fetch_scalar(self, sql: str) -> Any:
        session = await get_session()
        return await run(
            session.fetch_scalar,
            sql,
            table=self.physical_table_name,
        )


__all__ = ["SeekdbRepoBase"]
