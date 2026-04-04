"""
Episodic Memory Milvus Repository

Specialized repository class for episodic memory based on BaseMilvusRepository, providing efficient vector storage and retrieval functions.
Main features include vector storage, similarity search, filtered queries, and document management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import json
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository

logger = get_logger(__name__)

# Milvus retrieval configuration (None means radius filtering is disabled)
MILVUS_SIMILARITY_RADIUS = None  # COSINE similarity threshold, optional range [-1, 1]


@repository("episodic_memory_milvus_repository", primary=False)
class EpisodicMemoryMilvusRepository(BaseMilvusRepository[EpisodicMemoryCollection]):
    """
    Episodic Memory Milvus Repository

    Specialized repository class based on BaseMilvusRepository, providing:
    - Efficient vector storage and retrieval
    - Similarity search and filtering functions
    - Document creation and management
    - Vector index management
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
        user_name: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        linked_entities: Optional[List[str]] = None,
        subject: Optional[str] = None,
        memcell_event_id_list: Optional[List[str]] = None,
        extend: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create and save episodic memory document

        Args:
            event_id: Event unique identifier
            user_id: User ID (required)
            timestamp: Event occurrence time (required)
            episode: Episode description (required)
            search_content: List of search content (required)
            vector: Text vector (required, dimension must be VECTORIZE_DIMENSIONS)
            user_name: User name
            title: Event title
            summary: Event summary
            group_id: Group ID
            participants: List of participants
            event_type: Event type (e.g., conversation, email, etc.)
            keywords: List of keywords
            linked_entities: List of linked entity IDs
            subject: Event subject
            memcell_event_id_list: List of memory cell event IDs
            extend: Extension fields
            created_at: Creation time
            updated_at: Update time
            parent_event_id: Parent event ID (used to associate split records)
            metadata: Metadata JSON string (optional, automatically constructed if not provided)

        Returns:
            Saved document information
        """
        try:
            # Set default timestamps
            now = get_now_with_timezone()
            if created_at is None:
                created_at = now
            if updated_at is None:
                updated_at = now

            # Prepare metadata (automatically build if not provided externally)
            if metadata is None:
                metadata_dict = {
                    "user_name": user_name or "",
                    "title": title or "",
                    "summary": summary or "",
                    "participants": participants or [],
                    "keywords": keywords or [],
                    "linked_entities": linked_entities or [],
                    "subject": subject or "",
                    "memcell_event_id_list": memcell_event_id_list or [],
                    "extend": extend or {},
                    "created_at": created_at.isoformat(),
                    "updated_at": updated_at.isoformat(),
                }
                metadata_json = json.dumps(metadata_dict, ensure_ascii=False)
            else:
                # Use externally provided metadata
                metadata_json = metadata
                try:
                    metadata_dict = json.loads(metadata)
                except:
                    metadata_dict = {}

            # Prepare entity data
            entity = {
                "id": id,
                "vector": vector,
                "user_id": user_id
                or "",  # Milvus VARCHAR does not accept None, convert to empty string
                "group_id": group_id or "",
                "participants": participants or [],
                "parent_type": parent_type or "",
                "parent_id": parent_id or "",
                "event_type": event_type or "",
                "timestamp": int(timestamp.timestamp()),
                "episode": episode,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "metadata": metadata_json,
                "created_at": int(created_at.timestamp()),
                "updated_at": int(updated_at.timestamp()),
            }

            # Insert data
            await self.insert(entity)

            logger.debug(
                "✅ Episodic memory document created successfully: id=%s, user_id=%s",
                id,
                user_id,
            )

            return {
                "id": id,
                "user_id": user_id,
                "timestamp": timestamp,
                "episode": episode,
                "search_content": search_content,
                "metadata": metadata_dict,
            }

        except Exception as e:
            logger.error(
                "❌ Failed to create episodic memory document: id=%s, error=%s", id, e
            )
            raise

    # ==================== Search Functionality ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
        participant_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search

        Args:
            query_vector: Query vector
            user_id: User ID filter
            group_id: Group ID filter
            event_type: Event type filter (e.g., conversation, email, etc.)
            start_time: Start timestamp filter
            end_time: End timestamp filter
            limit: Number of results to return
            score_threshold: Similarity threshold
            radius: COSINE similarity threshold (optional, defaults to MILVUS_SIMILARITY_RADIUS)

        Returns:
            List of search results
        """
        try:
            # Build filter expression
            filter_expr = []

            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL:
                if user_id:
                    filter_expr.append(f'user_id == "{user_id}"')
                else:
                    # Explicitly filter for null or empty
                    filter_expr.append('user_id == ""')

            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL:
                if group_id:
                    filter_expr.append(f'group_id == "{group_id}"')
                else:
                    # Explicitly filter for null or empty
                    filter_expr.append('group_id == ""')

            if participant_user_id:
                filter_expr.append(
                    f'array_contains(participants, "{participant_user_id}")'
                )
            if event_type:
                filter_expr.append(f'event_type == "{event_type}"')
            if start_time:
                filter_expr.append(f'timestamp >= {int(start_time.timestamp())}')
            if end_time:
                filter_expr.append(f'timestamp <= {int(end_time.timestamp())}')

            filter_str = " and ".join(filter_expr) if filter_expr else None

            # Get collection

            # Execute search
            # Dynamically adjust ef parameter: must be >= limit, typically set to 1.5-2 times limit
            ef_value = max(128, limit * 2)  # Ensure ef >= limit, minimum 128
            # Use COSINE similarity, radius indicates returning only results with similarity >= threshold
            # Prioritize passed radius parameter, otherwise use default configuration
            similarity_radius = (
                radius if radius is not None else MILVUS_SIMILARITY_RADIUS
            )
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}
            # Do not set radius parameter!
            # Milvus radius is the similarity lower bound; setting too low a value may cause issues
            # Only set when explicitly specified and > -1.0
            if similarity_radius is not None and similarity_radius > -1.0:
                search_params["params"]["radius"] = similarity_radius

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
            raw_hit_count = sum(len(hits) for hits in results)
            logger.info(
                f"Milvus raw return: {raw_hit_count} results, "
                f"limit={limit}, filter_str={filter_str}, "
            )

            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        # Parse metadata
                        metadata_json = hit.entity.get("metadata", "{}")
                        metadata = json.loads(metadata_json) if metadata_json else {}

                        # Parse search_content (unified as JSON array format)
                        search_content_raw = hit.entity.get("search_content", "[]")
                        search_content = (
                            json.loads(search_content_raw) if search_content_raw else []
                        )

                        result = {
                            "id": hit.entity.get("id"),
                            "score": float(hit.score),
                            "user_id": hit.entity.get("user_id"),
                            "group_id": hit.entity.get("group_id"),
                            "event_type": hit.entity.get("event_type"),
                            "timestamp": datetime.fromtimestamp(
                                hit.entity.get("timestamp", 0)
                            ),
                            "episode": hit.entity.get("episode"),
                            "search_content": search_content,
                            "metadata": metadata,
                        }
                        search_results.append(result)

            logger.debug(
                "✅ Vector search successful: Found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("❌ Vector search failed: %s", e)
            raise

    # ==================== Deletion Functionality ====================

    async def delete_by_parent_id(
        self,
        parent_id: str,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Delete episodic memory vectors by parent MemCell ID.

        Args:
            parent_id: Source MemCell ID
            user_id: If provided, only delete records for this user.

        Returns:
            Number of deleted records
        """
        try:
            expr = f'parent_id == "{parent_id}"'
            if user_id is not None:
                expr += f' and user_id == "{user_id}"'

            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)
            if delete_count > 0:
                await self.collection.delete(expr)
            logger.debug(
                "✅ Deleted episodic memory vectors by parent_id=%s user_id=%s: %d records",
                parent_id,
                user_id,
                delete_count,
            )
            return delete_count
        except Exception as e:
            logger.error(
                "❌ Failed to delete episodic memory vectors by parent_id=%s: %s",
                parent_id,
                e,
            )
            raise

    async def delete_by_event_id(self, event_id: str) -> bool:
        """
        Delete episodic memory document by event_id

        Args:
            event_id: Event unique identifier

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            success = await self.delete_by_id(event_id)
            if success:
                logger.debug(
                    "✅ Deleted episodic memory by event_id: event_id=%s", event_id
                )
            return success
        except Exception as e:
            logger.error(
                "❌ Failed to delete episodic memory by event_id: event_id=%s, error=%s",
                event_id,
                e,
            )
            return False

    async def delete_by_filters(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Batch delete episodic memory documents based on filter conditions

        Args:
            user_id: User ID filter
            group_id: Group ID filter
            start_time: Start time
            end_time: End time

        Returns:
            Number of deleted documents
        """
        try:
            # Build filter expression
            filter_expr = []
            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL and user_id:
                filter_expr.append(f'user_id == "{user_id}"')
            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL and group_id:
                filter_expr.append(f'group_id == "{group_id}"')
            if start_time:
                filter_expr.append(f'timestamp >= {int(start_time.timestamp())}')
            if end_time:
                filter_expr.append(f'timestamp <= {int(end_time.timestamp())}')

            if not filter_expr:
                raise ValueError("At least one filter condition must be provided")

            expr = " and ".join(filter_expr)

            # First query the number of documents to delete
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            # Execute deletion
            await self.collection.delete(expr)

            logger.debug(
                "✅ Batch deletion of episodic memory by filter conditions successful: Deleted %d records",
                delete_count,
            )
            return delete_count

        except Exception as e:
            logger.error(
                "❌ Batch deletion of episodic memory by filter conditions failed: %s",
                e,
            )
            raise
