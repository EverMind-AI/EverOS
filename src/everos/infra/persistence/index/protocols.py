"""Typed ports for the rebuildable derived-index subsystem."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from .predicate import Predicate

T_co = TypeVar("T_co", bound=BaseModel, covariant=True)
T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class IndexRepository(Protocol[T]):
    """Backend-neutral repository contract used by memory and cascade.

    Maintenance methods are deliberately part of the contract. Embedded
    engines can perform physical work; service-managed engines implement
    successful no-ops. This keeps scheduling and health semantics identical
    without making the cascade worker know which backend is active.
    """

    schema: type[T]

    @property
    def table_name(self) -> str: ...

    async def add(self, records: Sequence[T]) -> None: ...

    async def upsert(self, records: Sequence[T], *, by: str = "id") -> None: ...

    async def count(self) -> int: ...

    async def count_where(self, where: Predicate | None = None) -> int: ...

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> T | None: ...

    async def find_where(self, where: Predicate, *, limit: int = 100) -> list[T]: ...

    async def find_one_where(self, where: Predicate) -> T | None: ...

    async def find_by_owner(self, owner_id: str, *, limit: int = 100) -> list[T]: ...

    async def find_by_md_path(self, md_path: str) -> T | None: ...

    async def find_by_owner_entry(
        self,
        owner_id: str,
        entry_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> T | None: ...

    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[T]: ...

    async def find_by_session(
        self, owner_id: str, session_id: str, *, limit: int = 100
    ) -> list[T]: ...

    async def find_by_parent(
        self, parent_type: str, parent_id: str, *, limit: int = 100
    ) -> list[T]: ...

    async def find_where_paginated(
        self,
        where: Predicate,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20_000,
    ) -> tuple[list[T], int]: ...

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: Predicate | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Predicate | None,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Predicate | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]: ...

    async def scan(self, where: Predicate | None = None) -> list[T]: ...

    async def update(self, updates: dict[str, Any], *, where: Predicate) -> None: ...

    async def delete(self, predicate: Predicate) -> None: ...

    async def delete_by_md_path(self, md_path: str) -> int: ...

    async def optimize(self) -> None: ...

    async def prune(self, older_than: dt.timedelta) -> None: ...

    async def rebuild_indexes(self) -> None: ...


@runtime_checkable
class EpisodeIndexRepository(IndexRepository[T], Protocol[T]):
    """Episode-only reads every backend must also provide."""

    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int: ...

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
    ) -> list[Any]: ...


@runtime_checkable
class AgentSkillIndexRepository(IndexRepository[T], Protocol[T]):
    """AgentSkill-only cluster reads every backend must also provide."""

    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int: ...

    async def find_in_cluster(
        self, *, owner_id: str, cluster_id: str, limit: int
    ) -> list[Any]: ...

    async def find_topk_relevant_in_cluster(
        self,
        *,
        owner_id: str,
        cluster_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[Any]: ...


@runtime_checkable
class IndexBackend(Protocol):
    """Lifecycle contract for one configured derived-index backend."""

    name: str
    repositories: tuple[IndexRepository[Any], ...]

    async def connect(self) -> object: ...

    async def startup(self) -> object: ...

    async def shutdown(self) -> None: ...

    async def ensure_business_indexes(self) -> None: ...

    async def verify_business_schemas(self) -> None: ...

    async def drop_business_tables(self) -> list[str]: ...


__all__ = [
    "AgentSkillIndexRepository",
    "EpisodeIndexRepository",
    "IndexBackend",
    "IndexRepository",
]
