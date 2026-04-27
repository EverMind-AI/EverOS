"""
User Profile Milvus Repository

V1 simplified repository for vector semantic retrieval.
Only maps search-essential fields. Full data retrieved from MongoDB using id.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.user_profile_collection import (
    UserProfileCollection,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("user_profile_milvus_repository", primary=False)
class UserProfileMilvusRepository(BaseMilvusRepository[UserProfileCollection]):
    """
    User Profile Milvus Repository

    V1 simplified repository for vector semantic retrieval.
    Only stores search-essential fields in Milvus.
    Full data is retrieved from MongoDB using id.
    """

    def __init__(self):
        """Initialize user profile repository"""
        super().__init__(UserProfileCollection)

    # ==================== Search Functionality ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        scenario: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search for user profiles

        Args:
            query_vector: Query vector
            user_id: User ID filter
            group_id: Group ID filter
            scenario: Scenario type filter (solo/team)
            limit: Number of results to return
            score_threshold: Similarity score threshold
            radius: COSINE similarity threshold

        Returns:
            List of search results
        """
        try:
            # Build filter expression
            filter_expr = []

            # Handle user_id filter
            if user_id and user_id != MAGIC_ALL:
                filter_expr.append(f'user_id == "{user_id}"')

            # Handle group_id filter
            if group_id:
                filter_expr.append(f'group_id == "{group_id}"')

            # Handle scenario filter
            if scenario:
                filter_expr.append(f'scenario == "{scenario}"')

            filter_str = " and ".join(filter_expr) if filter_expr else None

            # Execute search
            ef_value = max(128, limit * 2)
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}

            if radius is not None and radius > -1.0:
                search_params["params"]["radius"] = radius

            results = await self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filter_str,
                output_fields=self.all_output_fields,
            )

            # Process results
            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        result = {
                            "id": hit.entity.get("id"),
                            "score": float(hit.score),
                            "user_id": hit.entity.get("user_id"),
                            "group_id": hit.entity.get("group_id"),
                            "scenario": hit.entity.get("scenario"),
                            "memcell_count": hit.entity.get("memcell_count"),
                            "item_type": hit.entity.get("item_type", ""),
                            "embed_text": hit.entity.get("embed_text", ""),
                        }
                        search_results.append(result)

            logger.debug(
                "Vector search succeeded: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("Vector search failed: %s", e)
            raise

    # ==================== Deletion Functionality ====================

    async def delete_by_user_group(self, user_id: str, group_id: str) -> int:
        """
        Delete all profiles for a user in a group

        Args:
            user_id: User ID
            group_id: Group ID

        Returns:
            Number of deleted entities
        """
        try:
            filter_expr = f'user_id == "{user_id}" and group_id == "{group_id}"'

            result = await self.collection.delete(filter_expr)

            count = result.delete_count if hasattr(result, 'delete_count') else 0

            logger.info(
                "Deleted profile items: user_id=%s, group_id=%s, count=%d",
                user_id,
                group_id,
                count,
            )

            return count

        except Exception as e:
            logger.error(
                "Failed to delete profile items: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            return 0
