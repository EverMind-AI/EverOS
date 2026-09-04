"""Milvus repo singletons for EverOS derived index tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from everos.component.utils.datetime import from_timestamp
from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
)
from everos.infra.persistence.predicate import all_of, eq, gt, is_null

from .repository import MilvusRepoBase


class _EpisodeRepo(MilvusRepoBase[Episode]):
    schema = Episode

    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int:
        return await self._count_where(
            all_of(
                eq("owner_id", owner_id),
                eq("app_id", app_id),
                eq("project_id", project_id),
                is_null("deprecated_by"),
                eq("parent_type", parent_type) if parent_type is not None else None,
            )
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
    ) -> list[Episode] | list[dict[str, Any]]:
        predicate = all_of(
            eq("owner_id", owner_id),
            gt("timestamp", from_timestamp(after_ts)),
            eq("parent_type", parent_type),
            eq("app_id", app_id),
            eq("project_id", project_id),
            is_null("deprecated_by"),
        )
        # A plain query(limit=...) cannot cross Milvus' 16,384-row window, so
        # the ceiling has to be applied through the iterator instead.
        raw = await self._scan_raw(
            predicate, include_vectors=True, max_rows=limit or 20_000
        )
        rows = [self._model_from_milvus(row) for row in raw]
        rows.sort(key=lambda row: row.timestamp)
        if columns is None:
            return rows
        projection = list(dict.fromkeys([*columns, "timestamp"]))
        return [{name: getattr(row, name) for name in projection} for row in rows]


class _AtomicFactRepo(MilvusRepoBase[AtomicFact]):
    schema = AtomicFact


class _ForesightRepo(MilvusRepoBase[Foresight]):
    schema = Foresight


class _AgentCaseRepo(MilvusRepoBase[AgentCase]):
    schema = AgentCase


class _AgentSkillRepo(MilvusRepoBase[AgentSkill]):
    schema = AgentSkill

    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int:
        return await self._count_where(
            all_of(eq("owner_id", owner_id), eq("cluster_id", cluster_id))
        )

    async def find_in_cluster(
        self, *, owner_id: str, cluster_id: str, limit: int
    ) -> list[AgentSkill]:
        return await self.find_where(
            all_of(eq("owner_id", owner_id), eq("cluster_id", cluster_id)),
            limit=limit,
        )

    async def find_topk_relevant_in_cluster(
        self,
        *,
        owner_id: str,
        cluster_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[AgentSkill]:
        if not query_vector:
            raise ValueError(
                "query_vector must be non-empty; "
                "call find_in_cluster for the scalar fallback"
            )
        rows = await self.dense_search(
            query_vector,
            all_of(eq("owner_id", owner_id), eq("cluster_id", cluster_id)),
            limit=top_k,
        )
        out: list[AgentSkill] = []
        for row in rows:
            rid = row.get("id")
            if isinstance(rid, str) and (item := await self.get_by_id(rid)) is not None:
                out.append(item)
        return out


class _UserProfileRepo(MilvusRepoBase[UserProfile]):
    schema = UserProfile


class _KnowledgeTopicRepo(MilvusRepoBase[KnowledgeTopic]):
    schema = KnowledgeTopic


episode_repo = _EpisodeRepo()
atomic_fact_repo = _AtomicFactRepo()
foresight_repo = _ForesightRepo()
agent_case_repo = _AgentCaseRepo()
agent_skill_repo = _AgentSkillRepo()
user_profile_repo = _UserProfileRepo()
knowledge_topic_repo = _KnowledgeTopicRepo()

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
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "episode_repo",
    "foresight_repo",
    "knowledge_topic_repo",
    "user_profile_repo",
]
