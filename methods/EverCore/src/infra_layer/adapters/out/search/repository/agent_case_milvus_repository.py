"""
AgentCase Milvus Repository

Provides vector search for agent case records via Milvus.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.agent_case_collection import (
    AgentCaseCollection,
)
from core.observation.logger import get_logger
from core.di.decorators import repository


logger = get_logger(__name__)

MILVUS_SIMILARITY_RADIUS = None


@repository("agent_case_milvus_repository", primary=False)
class AgentCaseMilvusRepository(BaseMilvusRepository[AgentCaseCollection]):
    """
    AgentCase Milvus Repository

    Supports vector similarity search over agent task-solving experiences.
    Filters: user_id, group_ids, timestamp range, parent_id.
    """

    def __init__(self):
        super().__init__(AgentCaseCollection)

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        parent_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search over agent cases.

        Args:
            query_vector: Query embedding vector
            user_id: User ID filter (MAGIC_ALL to skip)
            session_id: Session ID filter
            group_ids: Group ID list filter (None to skip)
            start_time: Filter records with timestamp >= start_time
            end_time: Filter records with timestamp <= end_time
            parent_id: Filter by parent MemCell ID
            limit: Maximum results to return
            score_threshold: Minimum COSINE similarity score
            radius: Explicit COSINE similarity threshold

        Returns:
            List of search result dicts
        """
        try:
            filter_expr = []

            if user_id != MAGIC_ALL:
                if user_id:
                    filter_expr.append(f'user_id == "{user_id}"')
                else:
                    filter_expr.append('user_id == ""')

            if session_id:
                filter_expr.append(f'session_id == "{session_id}"')

            if group_ids is not None and len(group_ids) > 0:
                group_ids_str = ", ".join(f'"{g}"' for g in group_ids)
                filter_expr.append(f"group_id in [{group_ids_str}]")

            if parent_id:
                filter_expr.append(f'parent_id == "{parent_id}"')

            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp())}")

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
                            "user_id": hit.entity.get("user_id"),
                            "group_id": hit.entity.get("group_id"),
                            "session_id": hit.entity.get("session_id", ""),
                            "timestamp": datetime.fromtimestamp(
                                hit.entity.get("timestamp", 0), tz=timezone.utc
                            ),
                            "task_intent": hit.entity.get("task_intent", ""),
                            "parent_type": hit.entity.get("parent_type", ""),
                            "parent_id": hit.entity.get("parent_id", ""),
                        }
                        search_results.append(result)

            logger.debug(
                "AgentCase vector search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("AgentCase vector search failed: %s", e)
            raise
