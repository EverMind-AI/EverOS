"""The neutral index contract, spoken natively by the LanceDB repositories.

:class:`LanceIndexContract` is mixed into every per-table repository next to
its chassis base (``LanceRepoBase`` / ``LanceDailyLogRepoBase``). The chassis
speaks LanceDB's SQL-like ``where`` strings; the contract
(:mod:`everos.infra.persistence.index.protocols`) speaks the neutral
:class:`Predicate`. The mixin renders one into the other on the way down and
adds the reads the chassis does not have (``count_where``, ``scan``, the two
search routes). Everything else — ``upsert`` merging on the declared
``PRIMARY_KEY``, maintenance, the ``find_by_*`` helpers — is the chassis
method itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from collections.abc import Sequence
from typing import Any, Final

from lancedb.query import BooleanQuery, FullTextQuery, MatchQuery

try:
    from lancedb.query import Occur
except ImportError:  # pragma: no cover
    from lancedb._lancedb import Occur  # type: ignore[attr-defined,no-redef]

from everos.component.utils.datetime import ensure_utc, to_iso_format
from everos.core.persistence.lancedb.base import VECTOR_QUERY_NPROBES
from everos.infra.persistence.predicate import (
    All,
    AnyOf,
    Comparison,
    Contains,
    In,
    IsNull,
    Predicate,
    Scalar,
    all_of,
    eq,
    one_of,
)

_FIELD_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPERATORS: Final[dict[str, str]] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


def render_predicate(predicate: Predicate | None) -> str:
    """Render a neutral predicate as a LanceDB DataFusion expression."""
    if predicate is None:
        return ""
    if isinstance(predicate, Comparison):
        return (
            f"{_field(predicate.field)} {_OPERATORS[predicate.operator]} "
            f"{_literal(predicate.value)}"
        )
    if isinstance(predicate, In):
        values = ", ".join(_literal(value) for value in predicate.values)
        return f"{_field(predicate.field)} IN ({values})"
    if isinstance(predicate, Contains):
        return f"array_has({_field(predicate.field)}, {_literal(predicate.value)})"
    if isinstance(predicate, IsNull):
        return f"{_field(predicate.field)} IS NULL"
    if isinstance(predicate, All):
        return _render_group(predicate.children, "AND")
    if isinstance(predicate, AnyOf):
        return _render_group(predicate.children, "OR")
    raise TypeError(f"unsupported predicate: {type(predicate).__name__}")


def _render_group(children: tuple[Predicate, ...], operator: str) -> str:
    rendered = [render_predicate(child) for child in children]
    rendered = [item for item in rendered if item]
    if not rendered:
        return ""
    if len(rendered) == 1:
        return rendered[0]
    return "(" + f" {operator} ".join(f"({item})" for item in rendered) + ")"


def _field(value: str) -> str:
    if not _FIELD_NAME.fullmatch(value):
        raise ValueError(f"invalid predicate field: {value!r}")
    return value


def _literal(value: Scalar) -> str:
    if isinstance(value, str):
        return f"'{value.replace(chr(39), chr(39) * 2)}'"
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return f"TIMESTAMP '{to_iso_format(aware)}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _render_optional(predicate: Predicate | None) -> str | None:
    rendered = render_predicate(predicate)
    return rendered or None


def _where(where: Predicate | str) -> str:
    """The chassis' own callers pass strings; the contract passes predicates."""
    return where if isinstance(where, str) else render_predicate(where)


def build_or_query(tokens: Sequence[str], column: str) -> FullTextQuery | None:
    """Build LanceDB's OR-mode BM25 query for already-tokenized terms."""
    clean = [token for token in tokens if token]
    if not clean:
        return None
    if len(clean) == 1:
        return MatchQuery(clean[0], column=column)
    return BooleanQuery(
        [(Occur.SHOULD, MatchQuery(token, column=column)) for token in clean]
    )


class LanceIndexContract[T]:
    """Accept the neutral predicate on the chassis' string-where methods."""

    # ``schema``, ``table_name`` and ``_table()`` come from the chassis base that
    # follows this mixin in the MRO; nothing is declared here so nothing shadows it.

    async def count_where(self, where: Predicate | None = None) -> int:
        table = await self._table()
        return await table.count_rows(filter=_render_optional(where))

    async def find_where(self, where: Predicate | str, *, limit: int = 100) -> list[T]:
        return await super().find_where(_where(where), limit=limit)  # type: ignore[misc]

    async def find_one_where(self, where: Predicate | str) -> T | None:
        return await super().find_one_where(_where(where))  # type: ignore[misc]

    async def find_where_paginated(
        self,
        where: Predicate | str,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20_000,
    ) -> tuple[list[T], int]:
        return await super().find_where_paginated(  # type: ignore[misc]
            _where(where),
            sort_by=sort_by,
            descending=descending,
            page=page,
            page_size=page_size,
            max_fetch=max_fetch,
        )

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: Predicate | str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if vector is not None:
            # Route through dense_search so the metric is pinned to cosine.
            # The chassis search leaves distance_type at the table default
            # (L2), which would not match the Milvus backend's COSINE.
            return await self.dense_search(vector, where, limit=limit)
        rendered = where if isinstance(where, str) else _render_optional(where)
        return await super().search(where=rendered, limit=limit)  # type: ignore[misc]

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Predicate | str | None,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        fields = list(columns or getattr(self.schema, "BM25_FIELDS", ()))
        clean_terms = [term for term in query_terms if term]
        if not clean_terms or not fields:
            return []
        table = await self._table()
        expression = where if isinstance(where, str) else _render_optional(where)

        async def _query_one(field: str) -> list[dict[str, Any]]:
            query = build_or_query(clean_terms, field)
            assert query is not None
            builder = table.query().nearest_to_text(query)
            if expression:
                builder = builder.where(expression)
            return await builder.limit(limit).to_list()

        per_column = await asyncio.gather(*(_query_one(field) for field in fields))
        best: dict[str, dict[str, Any]] = {}
        for rows in per_column:
            for row in rows:
                rid = row.get("id")
                if not isinstance(rid, str):
                    continue
                score = float(row.get("_score", 0.0))
                prior = best.get(rid)
                if prior is None or score > float(prior.get("_score", 0.0)):
                    shaped = dict(row)
                    shaped["_score"] = score
                    best[rid] = shaped
        return sorted(
            best.values(),
            key=lambda row: float(row.get("_score", 0.0)),
            reverse=True,
        )[:limit]

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Predicate | str | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        if not vector:
            return []
        table = await self._table()
        builder = (
            table.query()
            .nearest_to(list(vector))
            .column(vector_field)
            .distance_type("cosine")
            .nprobes(VECTOR_QUERY_NPROBES)
        )
        expression = where if isinstance(where, str) else _render_optional(where)
        if expression:
            builder = builder.where(expression)
        return await builder.limit(limit).to_list()

    async def scan(self, where: Predicate | str | None = None) -> list[T]:
        """Return all matching records without a hidden row cap."""
        table = await self._table()
        builder = table.query()
        expression = where if isinstance(where, str) else _render_optional(where)
        if expression:
            builder = builder.where(expression)
        rows = await builder.to_list()
        return [self.schema.model_validate(row) for row in rows]

    # The daily-log chassis base has these four; the single-file tables do not,
    # and the contract asks every table for them. One predicate-based copy
    # serves all seven (the mixin sits first in the MRO).
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
        columns: Sequence[str] | None = None,
    ) -> list[T] | list[dict[str, Any]]:
        if not entry_ids:
            return []
        predicate = all_of(
            eq("owner_id", owner_id),
            one_of("entry_id", list(entry_ids)),
            eq("app_id", app_id),
            eq("project_id", project_id),
        )
        if columns is None:
            return await self.find_where(predicate, limit=len(entry_ids))
        # Column projection keeps the vector payload out of the read.
        table = await self._table()
        return await (
            table.query()
            .where(render_predicate(predicate))
            .select(list(columns))
            .limit(len(entry_ids))
            .to_list()
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

    async def update(self, updates: dict[str, Any], *, where: Predicate | str) -> None:
        await super().update(updates, where=_where(where))  # type: ignore[misc]

    async def delete(self, predicate: Predicate | str) -> None:
        await super().delete(_where(predicate))  # type: ignore[misc]


__all__ = ["LanceIndexContract", "build_or_query", "render_predicate"]
