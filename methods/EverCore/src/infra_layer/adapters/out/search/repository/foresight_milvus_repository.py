"""
Foresight Milvus Repository

V1 simplified repository for vector semantic retrieval.
Only maps search-essential fields. Full data retrieved from MongoDB using parent_id.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.foresight_collection import (
    ForesightCollection,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("foresight_milvus_repository", primary=False)
class ForesightMilvusRepository(BaseMilvusRepository[ForesightCollection]):
    """
    Foresight Milvus Repository

    V1 simplified repository for vector semantic retrieval.
    Only stores search-essential fields in Milvus.
    Full data is retrieved from MongoDB using parent_id.
    """

    def __init__(self):
        """Initialize foresight repository"""
        super().__init__(ForesightCollection)

    # ==================== Document Creation and Management ====================
    # TODO: add username
    async def create_and_save_foresight_mem(
        self,
        id: str,
        user_id: Optional[str],
        content: str,
        parent_id: str,
        parent_type: str,
        vector: List[float],
        group_id: Optional[str] = None,
        event_type: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
        search_content: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create and save personal foresight document

        Args:
            id: Unique identifier for foresight
            user_id: User ID (required)
            content: Foresight content (required)
            parent_id: Parent memory ID (required)
            parent_type: Parent memory type (memcell/episode)
            vector: Text vector (required)
            group_id: Group ID
            participants: List of related participants
            start_time: Foresight start time
            end_time: Foresight end time
            duration_days: Duration in days
            evidence: Evidence supporting this foresight
            search_content: List of searchable content
        Returns:
            Information of the saved document
        """
        try:
            # Build search content
            if search_content is None:
                search_content = [content]
                if evidence:
                    search_content.append(evidence)

            # Prepare entity data
            entity = {
                "id": id,
                "vector": vector,
                "user_id": user_id or "",
                "group_id": group_id or "",
                "participants": participants or [],
                "sender_ids": sender_ids or [],
                "parent_type": parent_type,
                "parent_id": parent_id,
                "start_time": int(start_time.timestamp() * 1000) if start_time else 0,
                "end_time": int(end_time.timestamp() * 1000) if end_time else 0,
                "duration_days": duration_days or 0,
                "content": content,
                "evidence": evidence or "",
                "type": event_type,
                "search_content": json.dumps(search_content, ensure_ascii=False),
            }

            # Insert data
            await self.insert(entity)

            logger.debug(
                "Created personal foresight document successfully: memory_id=%s, user_id=%s",
                id,
                user_id,
            )

            return {
                "id": id,
                "user_id": user_id,
                "content": content,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "search_content": search_content,
            }

        except Exception as e:
            logger.error(
                "Failed to create personal foresight document: id=%s, error=%s", id, e
            )
            raise

    # ==================== Search Functionality ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        sender_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search

        Args:
            query_vector: Query vector
            user_id: User ID filter
            group_ids: List of Group IDs to filter
            sender_id: Sender ID filter
            session_id: Session ID filter
            parent_type: Parent type filter
            parent_id: Parent memory ID filter
            start_time: Start timestamp filter
            end_time: End timestamp filter
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
            if user_id != MAGIC_ALL:
                if user_id:
                    filter_expr.append(f'user_id == "{user_id}"')
                else:
                    filter_expr.append('user_id == ""')

            # Handle group_ids filter
            if group_ids is not None and len(group_ids) > 0:
                group_ids_str = ', '.join(f'"{g}"' for g in group_ids)
                filter_expr.append(f'group_id in [{group_ids_str}]')

            # Handle sender_id filter (match against sender_ids array)
            if sender_id:
                filter_expr.append(f'array_contains(sender_ids, "{sender_id}")')

            # Handle session_id filter
            if session_id:
                filter_expr.append(f'session_id == "{session_id}"')

            # Handle parent_type filter
            if parent_type:
                filter_expr.append(f'parent_type == "{parent_type}"')

            # Handle parent_id filter
            if parent_id:
                filter_expr.append(f'parent_id == "{parent_id}"')

            # Handle time filters
            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp() * 1000)}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp() * 1000)}")

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
                            "sender_ids": hit.entity.get("sender_ids"),
                            "session_id": hit.entity.get("session_id"),
                            "participants": hit.entity.get("participants"),
                            "timestamp": hit.entity.get("timestamp"),
                            "parent_type": hit.entity.get("parent_type"),
                            "parent_id": hit.entity.get("parent_id"),
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

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Batch delete foresight documents by filter conditions

        Args:
            user_id: User ID filter
            group_id: Group ID filter
            start_time: Start time
            end_time: End time

        Returns:
            Number of deleted documents
        """
        try:
            filter_expr = []
            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL:
                if (
                    not user_id
                ):  # None or "" -> match empty string (null mapped to "" by converter)
                    filter_expr.append('user_id == ""')
                else:
                    filter_expr.append(f'user_id == "{user_id}"')
            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL:
                if not group_id:  # None or "" -> match empty string
                    filter_expr.append('group_id == ""')
                else:
                    filter_expr.append(f'group_id == "{group_id}"')
            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp() * 1000)}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp() * 1000)}")

            if not filter_expr:
                raise ValueError("At least one filter condition must be provided")

            expr = " and ".join(filter_expr)

            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            await self.collection.delete(expr)

            logger.debug("Batch deleted foresights: deleted %d records", delete_count)
            return delete_count

        except Exception as e:
            logger.error("Failed to batch delete foresights: %s", e)
            raise
