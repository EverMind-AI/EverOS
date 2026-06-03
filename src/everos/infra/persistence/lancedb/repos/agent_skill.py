"""LanceDB repo singleton for the ``agent_skill`` table."""

from __future__ import annotations

from collections.abc import Sequence

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceRepoBase

from ..lancedb_manager import get_table
from ..tables.agent_skill import AgentSkill


class _AgentSkillRepo(LanceRepoBase[AgentSkill]):
    schema = AgentSkill

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)

    async def count_in_cluster(self, *, owner_id: str, cluster_id: str) -> int:
        """Count skills under one ``(owner_id, cluster_id)``."""
        table = await self._table()
        return await table.count_rows(filter=_in_cluster(owner_id, cluster_id))

    async def find_in_cluster(
        self, *, owner_id: str, cluster_id: str, limit: int
    ) -> list[AgentSkill]:
        """Scalar fetch within one cluster; no ranking, capped at ``limit``."""
        return await self.find_where(_in_cluster(owner_id, cluster_id), limit=limit)

    async def find_topk_relevant_in_cluster(
        self,
        *,
        owner_id: str,
        cluster_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[AgentSkill]:
        """Top-K cosine-relevant skills inside one cluster.

        Cosine ranking is pushed down to LanceDB native ANN.
        ``distance_type("cosine")`` matches
        :class:`AgentSkillRecaller.dense_recall`, keeping ranking
        semantics consistent across read paths.

        Raises:
            ValueError: When ``query_vector`` is empty — no relevance
                signal is a caller-side policy decision; use
                :meth:`find_in_cluster` for the scalar fallback.
        """
        if not query_vector:
            raise ValueError(
                "query_vector must be non-empty; "
                "call find_in_cluster for the scalar fallback"
            )
        table = await self._table()
        rows = await (
            table.query()
            .nearest_to(list(query_vector))
            .distance_type("cosine")
            .where(_in_cluster(owner_id, cluster_id))
            .limit(top_k)
            .to_list()
        )
        # LanceDB appends ``_distance`` to ranked rows; strip it before
        # ``model_validate`` so this stays robust regardless of
        # pydantic ``extra`` mode on the schema.
        return [
            self.schema.model_validate({k: v for k, v in r.items() if k != "_distance"})
            for r in rows
        ]


def _q(value: str) -> str:
    """SQL single-quote escape for LanceDB ``where`` predicate literals."""
    return value.replace("'", "''")


def _in_cluster(owner_id: str, cluster_id: str) -> str:
    return f"owner_id = '{_q(owner_id)}' AND cluster_id = '{_q(cluster_id)}'"


agent_skill_repo = _AgentSkillRepo()
