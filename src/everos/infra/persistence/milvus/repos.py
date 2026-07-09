"""Milvus repo singletons for EverOS derived index tables."""

from __future__ import annotations

from collections.abc import Sequence

from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
)

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
        rows = await self.find_where(
            f"owner_id = '{_q(owner_id)}' AND cluster_id = '{_q(cluster_id)}'",
            limit=10_000,
        )
        return len(rows)

    async def find_in_cluster(
        self, *, owner_id: str, cluster_id: str, limit: int
    ) -> list[AgentSkill]:
        return await self.find_where(
            f"owner_id = '{_q(owner_id)}' AND cluster_id = '{_q(cluster_id)}'",
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
            f"owner_id = '{_q(owner_id)}' AND cluster_id = '{_q(cluster_id)}'",
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


def _q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


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
