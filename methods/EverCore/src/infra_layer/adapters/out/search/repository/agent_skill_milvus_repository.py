"""
AgentSkill Milvus Repository

Provides vector search for agent skill records via Milvus.
Supports cluster-level delete for the replace pattern used by AgentSkillExtractor.
"""

from typing import List, Optional, Dict, Any

from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.agent_skill_collection import (
    AgentSkillCollection,
)
from core.observation.logger import get_logger
from core.di.decorators import repository


logger = get_logger(__name__)

MILVUS_SIMILARITY_RADIUS = None


@repository("agent_skill_milvus_repository", primary=False)
class AgentSkillMilvusRepository(BaseMilvusRepository[AgentSkillCollection]):
    """
    AgentSkill Milvus Repository

    Supports vector similarity search over reusable skill items.
    Also supports cluster-level deletion to support the replace pattern.
    """

    def __init__(self):
        super().__init__(AgentSkillCollection)

    async def vector_search(
        self,
        query_vector: List[float],
        group_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
        maturity_threshold: Optional[float] = 0.6,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search over agent skill items.

        Args:
            query_vector: Query embedding vector
            group_ids: Group ID list filter (None to skip)
            user_id: User ID filter
            cluster_id: Filter by MemScene cluster ID
            limit: Maximum results to return
            score_threshold: Minimum COSINE similarity score
            radius: Explicit COSINE similarity threshold
            maturity_threshold: Minimum maturity score (0.0-1.0) to return.
                Set to None to include all skills regardless of maturity.
            confidence_threshold: Minimum confidence score (0.0-1.0) to return.
                Skills below this threshold are considered retired.
                Set to None to skip confidence filtering.

        Returns:
            List of search result dicts
        """
        try:
            filter_expr = []

            if maturity_threshold is not None:
                filter_expr.append(f"maturity_score >= {maturity_threshold}")

            if confidence_threshold is not None:
                filter_expr.append(f"confidence >= {confidence_threshold}")

            if user_id != MAGIC_ALL:
                if user_id:
                    filter_expr.append(f'user_id == "{user_id}"')
                else:
                    filter_expr.append('user_id == ""')

            if group_ids is not None and len(group_ids) > 0:
                group_ids_str = ", ".join(f'"{g}"' for g in group_ids)
                filter_expr.append(f"group_id in [{group_ids_str}]")

            if cluster_id:
                filter_expr.append(f'cluster_id == "{cluster_id}"')

            filter_str = " and ".join(filter_expr) if filter_expr else None

            ef_value = max(128, limit * 2)
            similarity_radius = (
                radius if radius is not None else MILVUS_SIMILARITY_RADIUS
            )
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}
            if radius is not None and radius > -1.0:
                search_params["params"]["radius"] = radius
            elif similarity_radius is not None and similarity_radius > -1.0:
                search_params["params"]["radius"] = similarity_radius

            results = await self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filter_str,
                output_fields=self.all_output_fields,
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        result = {
                            "id": hit.entity.get("id"),
                            "score": float(hit.score),
                            "user_id": hit.entity.get("user_id", ""),
                            "group_id": hit.entity.get("group_id"),
                            "cluster_id": hit.entity.get("cluster_id"),
                            "content": hit.entity.get("content", ""),
                        }
                        search_results.append(result)

            logger.debug(
                "AgentSkill vector search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("AgentSkill vector search failed: %s", e)
            raise

    async def delete_by_cluster_id(self, cluster_id: str) -> int:
        """
        Delete all Milvus records for a cluster (called before replacing skills).

        Args:
            cluster_id: MemScene cluster ID

        Returns:
            Number of deleted records
        """
        try:
            expr = f'cluster_id == "{cluster_id}"'
            existing = await self.collection.query(expr=expr, output_fields=["id"])
            count = len(existing)
            if count > 0:
                await self.collection.delete(expr)
                logger.debug(
                    "Deleted %d Milvus records for cluster=%s", count, cluster_id
                )
            return count
        except Exception as e:
            logger.error(
                "Failed to delete Milvus records for cluster=%s: %s", cluster_id, e
            )
            return 0
