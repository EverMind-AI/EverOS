"""LanceDB adapter for the backend-neutral derived-index ports."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from typing import Any, ClassVar, Final

from lancedb.query import BooleanQuery, FullTextQuery, MatchQuery
from pydantic import BaseModel

try:
    from lancedb.query import Occur
except ImportError:  # pragma: no cover
    from lancedb._lancedb import Occur  # type: ignore[attr-defined,no-redef]

from everos.component.utils.datetime import ensure_utc, to_iso_format
from everos.core.persistence import LanceRepoBase
from everos.infra.persistence import lancedb as _lancedb

from .predicate import (
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


class LanceIndexRepository[T: BaseModel]:
    """Translate the neutral repository contract onto an existing Lance repo."""

    def __init__(self, repo: LanceRepoBase[T], schema: type[T]) -> None:
        self._repo = repo
        self.schema = schema

    @property
    def table_name(self) -> str:
        return str(self.schema.TABLE_NAME)  # type: ignore[attr-defined]

    async def add(self, records: Sequence[T]) -> None:
        await self._repo.add(records)

    async def upsert(self, records: Sequence[T], *, by: str = "id") -> None:
        await self._repo.upsert(records, by=by)

    async def count(self) -> int:
        return await self._repo.count()

    async def count_where(self, where: Predicate | None = None) -> int:
        table = await _lancedb.get_table(self.table_name, self.schema)
        return await table.count_rows(filter=_render_optional(where))

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> T | None:
        return await self._repo.get_by_id(id_value, id_field=id_field)

    async def find_where(self, where: Predicate, *, limit: int = 100) -> list[T]:
        return await self._repo.find_where(render_predicate(where), limit=limit)

    async def find_one_where(self, where: Predicate) -> T | None:
        return await self._repo.find_one_where(render_predicate(where))

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
        return await self._repo.find_where_paginated(
            render_predicate(where),
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
        where: Predicate | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._repo.search(
            vector=vector,
            where=_render_optional(where),
            limit=limit,
        )

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Predicate | None,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        fields = list(columns or getattr(self.schema, "BM25_FIELDS", ()))
        clean_terms = [term for term in query_terms if term]
        if not clean_terms or not fields:
            return []
        table = await _lancedb.get_table(self.table_name, self.schema)
        best: dict[str, dict[str, Any]] = {}
        for field in fields:
            query = build_or_query(clean_terms, field)
            assert query is not None
            builder = table.query().nearest_to_text(query)
            expression = _render_optional(where)
            if expression:
                builder = builder.where(expression)
            rows = await builder.limit(limit).to_list()
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
        where: Predicate | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        if not vector:
            return []
        table = await _lancedb.get_table(self.table_name, self.schema)
        builder = (
            table.query()
            .nearest_to(list(vector))
            .column(vector_field)
            .distance_type("cosine")
        )
        expression = _render_optional(where)
        if expression:
            builder = builder.where(expression)
        return await builder.limit(limit).to_list()

    async def scan(self, where: Predicate | None = None) -> list[T]:
        """Return all matching records without a hidden row cap."""
        table = await _lancedb.get_table(self.table_name, self.schema)
        builder = table.query()
        expression = _render_optional(where)
        if expression:
            builder = builder.where(expression)
        rows = await builder.to_list()
        return [self.schema.model_validate(row) for row in rows]

    async def update(self, updates: dict[str, Any], *, where: Predicate) -> None:
        await self._repo.update(updates, where=render_predicate(where))

    async def delete(self, predicate: Predicate) -> None:
        await self._repo.delete(render_predicate(predicate))

    async def delete_by_md_path(self, md_path: str) -> int:
        return await self._repo.delete_by_md_path(md_path)

    async def optimize(self) -> None:
        await self._repo.optimize()

    async def prune(self, older_than: dt.timedelta) -> None:
        await self._repo.prune(older_than)

    async def rebuild_indexes(self) -> None:
        await self._repo.rebuild_indexes()

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


class LanceEpisodeRepository(LanceIndexRepository[_lancedb.Episode]):
    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int:
        return await _lancedb.episode_repo.count_by_owner(
            owner_id,
            app_id=app_id,
            project_id=project_id,
            parent_type=parent_type,
        )

    async def list_by_owner_after_ts(
        self,
        *,
        owner_id: str,
        after_ts: int,
        parent_type: str,
        app_id: str = "default",
        project_id: str = "default",
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[_lancedb.Episode] | list[dict[str, Any]]:
        return await _lancedb.episode_repo.list_by_owner_after_ts(
            owner_id=owner_id,
            after_ts=after_ts,
            parent_type=parent_type,
            app_id=app_id,
            project_id=project_id,
            columns=columns,
            limit=limit,
        )


class LanceAgentSkillRepository(LanceIndexRepository[_lancedb.AgentSkill]):
    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int:
        return await _lancedb.agent_skill_repo.count_in_cluster(
            owner_id=owner_id, cluster_id=cluster_id
        )

    async def find_in_cluster(
        self, *, owner_id: str, cluster_id: str, limit: int
    ) -> list[_lancedb.AgentSkill]:
        return await _lancedb.agent_skill_repo.find_in_cluster(
            owner_id=owner_id, cluster_id=cluster_id, limit=limit
        )

    async def find_topk_relevant_in_cluster(
        self,
        *,
        owner_id: str,
        cluster_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[_lancedb.AgentSkill]:
        return await _lancedb.agent_skill_repo.find_topk_relevant_in_cluster(
            owner_id=owner_id,
            cluster_id=cluster_id,
            query_vector=query_vector,
            top_k=top_k,
        )


episode_repo = LanceEpisodeRepository(_lancedb.episode_repo, _lancedb.Episode)
atomic_fact_repo = LanceIndexRepository(_lancedb.atomic_fact_repo, _lancedb.AtomicFact)
foresight_repo = LanceIndexRepository(_lancedb.foresight_repo, _lancedb.Foresight)
agent_case_repo = LanceIndexRepository(_lancedb.agent_case_repo, _lancedb.AgentCase)
agent_skill_repo = LanceAgentSkillRepository(
    _lancedb.agent_skill_repo, _lancedb.AgentSkill
)
user_profile_repo = LanceIndexRepository(
    _lancedb.user_profile_repo, _lancedb.UserProfile
)
knowledge_topic_repo = LanceIndexRepository(
    _lancedb.knowledge_topic_repo, _lancedb.KnowledgeTopic
)

ALL_REPOS = (
    episode_repo,
    atomic_fact_repo,
    foresight_repo,
    agent_case_repo,
    agent_skill_repo,
    user_profile_repo,
    knowledge_topic_repo,
)


class LanceIndexBackend:
    """Own LanceDB connection, schema, index, and repository lifecycle."""

    name: ClassVar[str] = "lancedb"
    repositories = ALL_REPOS

    async def connect(self) -> Any:
        return await _lancedb.get_connection()

    async def startup(self) -> Any:
        connection = await self.connect()
        await self.verify_business_schemas()
        await self.ensure_business_indexes()
        return connection

    async def shutdown(self) -> None:
        await _lancedb.dispose_connection()

    async def ensure_business_indexes(self) -> None:
        await _lancedb.ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        await _lancedb.verify_business_schemas()

    async def drop_business_tables(self) -> list[str]:
        return await _lancedb.drop_business_tables()


def _render_optional(predicate: Predicate | None) -> str | None:
    rendered = render_predicate(predicate)
    return rendered or None


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


lance_index_backend = LanceIndexBackend()

__all__ = [
    "ALL_REPOS",
    "LanceIndexBackend",
    "LanceIndexRepository",
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "build_or_query",
    "episode_repo",
    "foresight_repo",
    "knowledge_topic_repo",
    "lance_index_backend",
    "render_predicate",
    "user_profile_repo",
]
