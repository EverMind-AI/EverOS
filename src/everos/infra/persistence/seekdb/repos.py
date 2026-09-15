"""SeekDB repository singletons for all seven derived business indexes."""

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

from .codec import from_row
from .repository import SeekdbRepoBase


class _EpisodeRepo(SeekdbRepoBase[Episode]):
    schema = Episode

    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int:
        return await self.count_where(
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
        if columns is None:
            raw = await self._query_rows(
                predicate,
                columns=self._stored_columns(include_vectors=True),
                order_by="`timestamp_ms` ASC, `id` ASC",
                limit=limit,
            )
            return [self._model(row) for row in raw]
        projection = list(dict.fromkeys([*columns, "timestamp"]))
        fields = [self.index_schema.field(name) for name in projection]
        physical = [self._storage_name(field) for field in fields]
        raw = await self._query_rows(
            predicate,
            columns=physical,
            order_by="`timestamp_ms` ASC, `id` ASC",
            limit=limit,
        )
        result: list[dict[str, Any]] = []
        for row in raw:
            restored = from_row(row, self.index_schema)
            result.append({name: restored.get(name) for name in projection})
        return result


class _AtomicFactRepo(SeekdbRepoBase[AtomicFact]):
    schema = AtomicFact


class _ForesightRepo(SeekdbRepoBase[Foresight]):
    schema = Foresight


class _AgentCaseRepo(SeekdbRepoBase[AgentCase]):
    schema = AgentCase


class _AgentSkillRepo(SeekdbRepoBase[AgentSkill]):
    schema = AgentSkill

    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int:
        return await self.count_where(
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


class _UserProfileRepo(SeekdbRepoBase[UserProfile]):
    schema = UserProfile


class _KnowledgeTopicRepo(SeekdbRepoBase[KnowledgeTopic]):
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
