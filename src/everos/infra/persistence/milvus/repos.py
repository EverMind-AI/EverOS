"""Milvus repositories for EverOS derived index tables, one per table."""

from __future__ import annotations

from collections.abc import Sequence

from everos.infra.persistence.records import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
)
from everos.memory.ports.query import all_of, eq, primary_key

from .repository import MilvusRepoBase


class _EpisodeRepo(MilvusRepoBase[Episode]):
    schema = Episode


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
            if not isinstance(rid, str):
                continue
            # `id` repeats across (app_id, project_id) spaces: hydrate by the full key.
            item = await self.find_one_where(
                primary_key(str(row["app_id"]), str(row["project_id"]), rid)
            )
            if item is not None:
                out.append(item)
        return out


class _UserProfileRepo(MilvusRepoBase[UserProfile]):
    schema = UserProfile


class _KnowledgeTopicRepo(MilvusRepoBase[KnowledgeTopic]):
    schema = KnowledgeTopic
