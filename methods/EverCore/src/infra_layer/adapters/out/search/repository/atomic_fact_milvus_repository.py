"""
Atomic Fact Milvus Repository

V1 simplified repository for vector semantic retrieval.
Only maps search-essential fields. Full data retrieved from MongoDB using parent_id.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.atomic_fact_collection import (
    AtomicFactCollection,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("atomic_fact_milvus_repository", primary=False)
class AtomicFactMilvusRepository(BaseMilvusRepository[AtomicFactCollection]):
    """
    Atomic Fact Milvus Repository

    V1 simplified repository for vector semantic retrieval.
    Only stores search-essential fields in Milvus.
    Full data is retrieved from MongoDB using parent_id.
    """

    def __init__(self):
        """Initialize the atomic fact repository"""
        super().__init__(AtomicFactCollection)

    # ==================== Document Creation and Management ====================

    async def create_and_save_atomic_fact(
        self,
        id: str,
        user_id: Optional[str],
        atomic_fact: str,
        parent_id: str,
        parent_type: str,
        timestamp: datetime,
        vector: List[float],
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        search_content: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create and save an atomic fact document

        Args:
            id: Unique identifier for the atomic fact
            user_id: User ID (required)
            atomic_fact: Atomic fact content (required)
            parent_id: Parent memory ID (required)
            parent_type: Parent memory type (memcell/episode)
            timestamp: Event occurrence time (required)
            vector: Text vector (required, dimension must be VECTORIZE_DIMENSIONS)
            group_id: Group ID
            participants: List of related participants
            event_type: Event type (e.g., Conversation, Email, etc.)
            search_content: List of searchable content
        Returns:
            Information of the saved document
        """
        try:
            # Build search content
            if search_content is None:
                search_content = [atomic_fact]

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
                "type": event_type,
                "timestamp": int(timestamp.timestamp() * 1000),
                "atomic_fact": atomic_fact,
                "search_content": json.dumps(search_content, ensure_ascii=False),
            }

            # Insert data
            await self.insert(entity)

            logger.debug(
                "Successfully created atomic fact document: id=%s, user_id=%s",
                id,
                user_id,
            )

            return {
                "id": id,
                "user_id": user_id,
                "atomic_fact": atomic_fact,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "timestamp": timestamp,
                "search_content": search_content,
            }

        except Exception as e:
            logger.error(
                "Failed to create atomic fact document: id=%s, error=%s", id, e
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
                        }
                        search_results.append(result)

            logger.debug(
                "Vector search succeeded: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("Vector search failed: %s", e)
            raise

    async def batch_vector_search_by_parent_ids(
        self,
        query_vector: List[float],
        parent_ids: List[str],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Batch vector search AtomicFacts filtered by parent_ids.

        Used by MRAG Phase 3 to expand episodes into atomic facts.

        Args:
            query_vector: Query vector for similarity search
            parent_ids: List of parent IDs to filter by
            user_id: Optional user ID filter
            group_ids: Optional group IDs filter
            limit: Max results per parent (total limit = limit * len(parent_ids))
            score_threshold: Minimum similarity score

        Returns:
            List of search results with score
        """
        try:
            filter_expr = []

            # parent_id IN filter
            parent_ids_str = ", ".join(f'"{pid}"' for pid in parent_ids)
            filter_expr.append(f"parent_id in [{parent_ids_str}]")

            # user_id filter
            if user_id and user_id != MAGIC_ALL:
                filter_expr.append(f'user_id == "{user_id}"')

            # group_ids filter
            if group_ids:
                group_ids_str = ", ".join(f'"{g}"' for g in group_ids)
                filter_expr.append(f"group_id in [{group_ids_str}]")

            filter_str = " and ".join(filter_expr)

            total_limit = limit * len(parent_ids)
            ef_value = max(128, total_limit * 2)
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}

            results = await self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=total_limit,
                expr=filter_str,
                output_fields=self.all_output_fields,
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        search_results.append(
                            {
                                "id": hit.entity.get("id"),
                                "score": float(hit.score),
                                "user_id": hit.entity.get("user_id"),
                                "group_id": hit.entity.get("group_id"),
                                "parent_type": hit.entity.get("parent_type"),
                                "parent_id": hit.entity.get("parent_id"),
                                "atomic_fact": hit.entity.get("atomic_fact"),
                                "timestamp": hit.entity.get("timestamp"),
                                "participants": hit.entity.get("participants"),
                            }
                        )

            logger.debug(
                "Batch vector search by parent_ids succeeded: "
                "parent_ids=%d, results=%d",
                len(parent_ids),
                len(search_results),
            )
            return search_results

        except Exception as e:
            logger.error("Batch vector search by parent_ids failed: %s", e)
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
        Batch delete atomic fact documents by filter conditions

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

            logger.debug("Batch deleted atomic facts: deleted %d records", delete_count)
            return delete_count

        except Exception as e:
            logger.error("Failed to batch delete atomic facts: %s", e)
            raise
