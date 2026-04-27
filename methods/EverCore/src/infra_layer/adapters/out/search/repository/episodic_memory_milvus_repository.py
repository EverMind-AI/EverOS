"""
Episodic Memory Milvus Repository

V1 simplified repository for vector semantic retrieval.
Only maps search-essential fields. Full data retrieved from MongoDB using parent_id.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("episodic_memory_milvus_repository", primary=False)
class EpisodicMemoryMilvusRepository(BaseMilvusRepository[EpisodicMemoryCollection]):
    """
    Episodic Memory Milvus Repository

    V1 simplified repository for vector semantic retrieval.
    Only stores search-essential fields in Milvus.
    Full data is retrieved from MongoDB using parent_id.
    """

    def __init__(self):
        """Initialize episodic memory repository"""
        super().__init__(EpisodicMemoryCollection)

    # ==================== Document Creation and Management ====================

    async def create_and_save_episodic_memory(
        self,
        id: str,
        user_id: str,
        timestamp: datetime,
        episode: str,
        search_content: List[str],
        vector: List[float],
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        subject: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create and save episodic memory document

        Args:
            id: Event unique identifier
            user_id: User ID (required)
            timestamp: Event occurrence time (required)
            episode: Episode description (required)
            search_content: List of search content (required)
            vector: Text vector (required, dimension must be VECTORIZE_DIMENSIONS)
            title: Event title
            summary: Event summary
            group_id: Group ID
            participants: List of participants
            sender_ids: List of sender IDs
            event_type: Event type (e.g., conversation, email, etc.)
            subject: Event subject
            parent_type: Parent type
            parent_id: Parent ID (used to associate split records)

        Returns:
            Saved document information
        """
        try:
            # Prepare entity data
            entity = {
                "id": id,
                "vector": vector,
                "user_id": user_id
                or "",  # Milvus VARCHAR does not accept None, convert to empty string
                "group_id": group_id or "",
                "participants": participants or [],
                "sender_ids": sender_ids or [],
                "parent_type": parent_type or "",
                "parent_id": parent_id or "",
                "type": event_type or "",
                "timestamp": int(timestamp.timestamp() * 1000),
                "episode": episode,
                "search_content": json.dumps(search_content, ensure_ascii=False),
            }

            # Insert data
            await self.insert(entity)

            logger.debug(
                "Episodic memory document created successfully: id=%s, user_id=%s",
                id,
                user_id,
            )

            return {
                "id": id,
                "user_id": user_id,
                "timestamp": timestamp,
                "episode": episode,
                "search_content": search_content,
            }

        except Exception as e:
            logger.error(
                "Failed to create episodic memory document: id=%s, error=%s", id, e
            )
            raise

    # ==================== Search Functionality ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
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
                            "session_id": hit.entity.get("session_id"),
                            "participants": hit.entity.get("participants"),
                            "timestamp": hit.entity.get("timestamp"),
                            "parent_type": hit.entity.get("parent_type"),
                            "parent_id": hit.entity.get("parent_id"),
                            "type": hit.entity.get("type"),
                            "episode": hit.entity.get("episode"),
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
        Batch delete episodic memory documents by filter conditions

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

            logger.debug(
                "Batch deleted episodic memories: deleted %d records", delete_count
            )
            return delete_count

        except Exception as e:
            logger.error("Failed to batch delete episodic memories: %s", e)
            raise
