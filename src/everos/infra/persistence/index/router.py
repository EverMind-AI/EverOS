"""Route stable repository objects to the configured derived-index backend."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel

from everos.config import load_settings

from .predicate import Predicate
from .protocols import (
    AgentSkillIndexRepository,
    EpisodeIndexRepository,
    IndexRepository,
)


class RoutedIndexRepository[T: BaseModel]:
    """Stable repository identity with a backend selected at call time."""

    def __init__(
        self,
        lance_repo: IndexRepository[T],
        milvus_repo_name: str,
    ) -> None:
        self._lance_repo = lance_repo
        self._milvus_repo_name = milvus_repo_name
        self.schema = lance_repo.schema

    @property
    def table_name(self) -> str:
        return self._lance_repo.table_name

    def _repo(self) -> IndexRepository[T]:
        if load_settings().index.backend == "milvus":
            from everos.infra.persistence import milvus

            return getattr(milvus, self._milvus_repo_name)
        return self._lance_repo

    async def add(self, records: Sequence[T]) -> None:
        await self._repo().add(records)

    async def upsert(self, records: Sequence[T], *, by: str = "id") -> None:
        await self._repo().upsert(records, by=by)

    async def count(self) -> int:
        return await self._repo().count()

    async def count_where(self, where: Predicate | None = None) -> int:
        return await self._repo().count_where(where)

    async def get_by_id(self, id_value: str, *, id_field: str = "id") -> T | None:
        return await self._repo().get_by_id(id_value, id_field=id_field)

    async def find_where(self, where: Predicate, *, limit: int = 100) -> list[T]:
        return await self._repo().find_where(where, limit=limit)

    async def find_one_where(self, where: Predicate) -> T | None:
        return await self._repo().find_one_where(where)

    async def find_by_owner(self, owner_id: str, *, limit: int = 100) -> list[T]:
        return await self._repo().find_by_owner(owner_id, limit=limit)

    async def find_by_md_path(self, md_path: str) -> T | None:
        return await self._repo().find_by_md_path(md_path)

    async def find_by_owner_entry(
        self,
        owner_id: str,
        entry_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> T | None:
        return await self._repo().find_by_owner_entry(
            owner_id,
            entry_id,
            app_id=app_id,
            project_id=project_id,
        )

    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[T]:
        return await self._repo().find_by_owner_entries(
            owner_id,
            entry_ids,
            app_id=app_id,
            project_id=project_id,
        )

    async def find_by_session(
        self, owner_id: str, session_id: str, *, limit: int = 100
    ) -> list[T]:
        return await self._repo().find_by_session(
            owner_id,
            session_id,
            limit=limit,
        )

    async def find_by_parent(
        self, parent_type: str, parent_id: str, *, limit: int = 100
    ) -> list[T]:
        return await self._repo().find_by_parent(
            parent_type,
            parent_id,
            limit=limit,
        )

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
        return await self._repo().find_where_paginated(
            where,
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
        return await self._repo().search(vector=vector, where=where, limit=limit)

    async def sparse_search(
        self,
        query_terms: Sequence[str],
        where: Predicate | None,
        *,
        columns: Sequence[str] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await self._repo().sparse_search(
            query_terms,
            where,
            columns=columns,
            limit=limit,
        )

    async def dense_search(
        self,
        vector: Sequence[float],
        where: Predicate | None,
        *,
        limit: int,
        vector_field: str = "vector",
    ) -> list[dict[str, Any]]:
        return await self._repo().dense_search(
            vector,
            where,
            limit=limit,
            vector_field=vector_field,
        )

    async def scan(self, where: Predicate | None = None) -> list[T]:
        return await self._repo().scan(where)

    async def update(self, updates: dict[str, Any], *, where: Predicate) -> None:
        await self._repo().update(updates, where=where)

    async def delete(self, predicate: Predicate) -> None:
        await self._repo().delete(predicate)

    async def delete_by_md_path(self, md_path: str) -> int:
        return await self._repo().delete_by_md_path(md_path)

    async def optimize(self) -> None:
        await self._repo().optimize()

    async def prune(self, older_than: dt.timedelta) -> None:
        await self._repo().prune(older_than)

    async def rebuild_indexes(self) -> None:
        await self._repo().rebuild_indexes()


class RoutedEpisodeRepository(RoutedIndexRepository[Any]):
    def _repo(self) -> EpisodeIndexRepository[Any]:
        return cast(EpisodeIndexRepository[Any], super()._repo())

    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int:
        return await self._repo().count_by_owner(
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
    ) -> list[Any]:
        return await self._repo().list_by_owner_after_ts(
            owner_id=owner_id,
            after_ts=after_ts,
            parent_type=parent_type,
            app_id=app_id,
            project_id=project_id,
            columns=columns,
            limit=limit,
        )


class RoutedAgentSkillRepository(RoutedIndexRepository[Any]):
    def _repo(self) -> AgentSkillIndexRepository[Any]:
        return cast(AgentSkillIndexRepository[Any], super()._repo())

    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int:
        return await self._repo().count_in_cluster(
            owner_id=owner_id,
            cluster_id=cluster_id,
        )

    async def find_in_cluster(
        self, *, owner_id: str, cluster_id: str, limit: int
    ) -> list[Any]:
        return await self._repo().find_in_cluster(
            owner_id=owner_id,
            cluster_id=cluster_id,
            limit=limit,
        )

    async def find_topk_relevant_in_cluster(
        self,
        *,
        owner_id: str,
        cluster_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[Any]:
        return await self._repo().find_topk_relevant_in_cluster(
            owner_id=owner_id,
            cluster_id=cluster_id,
            query_vector=query_vector,
            top_k=top_k,
        )


__all__ = [
    "RoutedAgentSkillRepository",
    "RoutedEpisodeRepository",
    "RoutedIndexRepository",
]
