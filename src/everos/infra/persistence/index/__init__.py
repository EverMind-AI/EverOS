"""Backend-neutral derived index persistence facade.

Markdown is the source of truth and SQLite stores system state. This package
selects the rebuildable vector/BM25 index backend used by cascade, search, and
get. The default backend remains LanceDB; Milvus is opt-in through
``[index] backend = "milvus"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lancedb.query import BooleanQuery, FullTextQuery, MatchQuery

try:
    from lancedb.query import Occur
except ImportError:  # pragma: no cover
    from lancedb._lancedb import Occur  # type: ignore[attr-defined,no-redef]

from everos.config import load_settings
from everos.infra.persistence import lancedb as _lancedb
from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    ParentType,
    UserProfile,
)
from everos.infra.persistence.lancedb.predicate import render_predicate as _render_lance

from .predicate import (
    Predicate,
    all_of,
    any_of,
    contains,
    eq,
    gt,
    gte,
    is_null,
    lt,
    lte,
    ne,
    one_of,
)
from .schema import schema_for


def active_backend() -> str:
    """Return the configured derived index backend name."""
    return load_settings().index.backend


async def startup() -> Any:
    """Initialise the configured derived index backend."""
    if active_backend() == "milvus":
        milvus = _milvus()
        await milvus.get_client()
        await milvus.ensure_business_indexes()
        return "milvus"
    conn = await _lancedb.get_connection()
    await _lancedb.verify_business_schemas()
    await _lancedb.ensure_business_indexes()
    return conn


async def connect() -> Any:
    """Open the active backend without creating or verifying indexes."""
    if active_backend() == "milvus":
        return await _milvus().get_client()
    return await _lancedb.get_connection()


async def shutdown() -> None:
    """Dispose the configured derived index backend."""
    if active_backend() == "milvus":
        await _milvus().dispose_connection()
    else:
        await _lancedb.dispose_connection()


async def ensure_business_indexes() -> None:
    if active_backend() == "milvus":
        await _milvus().ensure_business_indexes()
    else:
        await _lancedb.ensure_business_indexes()


async def verify_business_schemas() -> None:
    if active_backend() == "milvus":
        await _milvus().verify_business_schemas()
    else:
        await _lancedb.verify_business_schemas()


async def drop_business_tables() -> list[str]:
    """Drop all business indexes for the active backend."""
    if active_backend() == "milvus":
        return await _milvus().drop_business_tables()
    return await _lancedb.drop_business_tables()


class _LanceIndexRepoAdapter:
    """Add backend-neutral recall helpers to an existing LanceDB repo."""

    def __init__(self, repo: Any, schema: type[Any]) -> None:
        self._repo = repo
        self.schema = schema
        self.index_schema = schema_for(schema)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repo, name)

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Any,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        fields = list(columns or self.index_schema.bm25_fields)
        if not query_terms or not fields:
            return []
        table = await _lancedb.get_table(self.schema.TABLE_NAME, self.schema)
        expr = _lance_expr(where)
        best: dict[str, dict[str, Any]] = {}
        for field in fields:
            query = _build_or_query(query_terms, field)
            rows = (
                await table.query()
                .nearest_to_text(query)
                .where(expr)
                .limit(limit)
                .to_list()
            )
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
            best.values(), key=lambda row: float(row.get("_score", 0.0)), reverse=True
        )[:limit]

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Any,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        if not vector:
            return []
        table = await _lancedb.get_table(self.schema.TABLE_NAME, self.schema)
        return (
            await table.query()
            .nearest_to(list(vector))
            .column(vector_field)
            .distance_type("cosine")
            .where(_lance_expr(where))
            .limit(limit)
            .to_list()
        )


class _IndexRepoRouter:
    """Route repo calls to the configured derived index backend."""

    def __init__(
        self, lance_repo: Any, milvus_repo_name: str, schema: type[Any]
    ) -> None:
        self._lance = _LanceIndexRepoAdapter(lance_repo, schema)
        self._milvus_repo_name = milvus_repo_name
        self.schema = schema
        self.index_schema = schema_for(schema)

    @property
    def table_name(self) -> str:
        return self.index_schema.table_name

    def _repo(self) -> Any:
        if active_backend() == "milvus":
            return getattr(_milvus(), self._milvus_repo_name)
        return self._lance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._repo(), name)

    async def add(self, records: Sequence[Any]) -> None:
        await self._repo().add(records)

    async def upsert(self, records: Sequence[Any], *, by: str = "id") -> None:
        await self._repo().upsert(records, by=by)

    async def count(self) -> int:
        return await self._repo().count()

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> Any:
        return await self._repo().get_by_id(id_value, id_field=id_field)

    async def find_where(self, where: Any, *, limit: int = 100) -> list[Any]:
        return await self._repo().find_where(_where_for_backend(where), limit=limit)

    async def find_one_where(self, where: Any) -> Any:
        return await self._repo().find_one_where(_where_for_backend(where))

    async def find_where_paginated(
        self,
        where: Any,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20_000,
    ) -> tuple[list[Any], int]:
        return await self._repo().find_where_paginated(
            _where_for_backend(where),
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
        where: Any = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._repo().search(
            vector=vector, where=_where_for_backend(where), limit=limit
        )

    async def update(self, updates: dict[str, Any], *, where: Any) -> None:
        await self._repo().update(updates, where=_where_for_backend(where))

    async def delete(self, predicate: Any) -> None:
        await self._repo().delete(_where_for_backend(predicate))

    async def delete_by_md_path(self, md_path: str) -> int:
        return await self._repo().delete_by_md_path(md_path)

    async def optimize(self, **kwargs: Any) -> None:
        await self._repo().optimize(**kwargs)

    async def rebuild_indexes(self) -> None:
        await self._repo().rebuild_indexes()

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Any,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await self._repo().sparse_search(
            query_terms,
            _where_for_backend(where),
            columns=columns,
            limit=limit,
        )

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Any,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        return await self._repo().dense_search(
            vector,
            _where_for_backend(where),
            limit=limit,
            vector_field=vector_field,
        )


def _where_for_backend(where: Any) -> Any:
    if where is None:
        return None if active_backend() == "milvus" else ""
    if not isinstance(where, Predicate):
        raise TypeError(
            "derived-index repository predicates must use the neutral "
            f"Predicate AST, got {type(where).__name__}"
        )
    if active_backend() == "milvus":
        return where
    return _render_lance(where)


def _lance_expr(where: Any) -> str:
    if isinstance(where, Predicate):
        return _render_lance(where)
    if isinstance(where, str):
        return where
    raise TypeError(f"unsupported LanceDB predicate: {type(where).__name__}")


def _build_or_query(tokens: Sequence[str], column: str) -> FullTextQuery:
    clean = [token for token in tokens if token]
    if not clean:
        return MatchQuery("", column=column)
    if len(clean) == 1:
        return MatchQuery(clean[0], column=column)
    return BooleanQuery(
        [(Occur.SHOULD, MatchQuery(token, column=column)) for token in clean]
    )


def _milvus() -> Any:
    from everos.infra.persistence import milvus

    return milvus


episode_repo = _IndexRepoRouter(_lancedb.episode_repo, "episode_repo", Episode)
atomic_fact_repo = _IndexRepoRouter(
    _lancedb.atomic_fact_repo, "atomic_fact_repo", AtomicFact
)
foresight_repo = _IndexRepoRouter(_lancedb.foresight_repo, "foresight_repo", Foresight)
agent_case_repo = _IndexRepoRouter(
    _lancedb.agent_case_repo, "agent_case_repo", AgentCase
)
agent_skill_repo = _IndexRepoRouter(
    _lancedb.agent_skill_repo, "agent_skill_repo", AgentSkill
)
user_profile_repo = _IndexRepoRouter(
    _lancedb.user_profile_repo, "user_profile_repo", UserProfile
)
knowledge_topic_repo = _IndexRepoRouter(
    _lancedb.knowledge_topic_repo, "knowledge_topic_repo", KnowledgeTopic
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

__all__ = [
    "ALL_REPOS",
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Episode",
    "Foresight",
    "KnowledgeTopic",
    "ParentType",
    "Predicate",
    "UserProfile",
    "active_backend",
    "agent_case_repo",
    "agent_skill_repo",
    "all_of",
    "any_of",
    "atomic_fact_repo",
    "connect",
    "contains",
    "drop_business_tables",
    "ensure_business_indexes",
    "episode_repo",
    "eq",
    "foresight_repo",
    "gt",
    "gte",
    "is_null",
    "knowledge_topic_repo",
    "lt",
    "lte",
    "ne",
    "one_of",
    "shutdown",
    "startup",
    "user_profile_repo",
    "verify_business_schemas",
]
