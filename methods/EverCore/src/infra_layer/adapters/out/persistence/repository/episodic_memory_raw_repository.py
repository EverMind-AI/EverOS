from datetime import datetime
from typing import List, Optional, Dict, Any, Type, TypeVar, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.mongo.mongo_utils import build_id_filter as _build_id_filter
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
    EpisodicMemoryProjection,
)

T = TypeVar("T")
from agentic_layer.vectorize_service import get_vectorize_service

logger = get_logger(__name__)


@repository("episodic_memory_raw_repository", primary=True)
class EpisodicMemoryRawRepository(BaseRepository[EpisodicMemory]):
    """
    Episodic memory raw data repository
    Generates vectorized text content and saves it to the database
    Provides CRUD operations and basic query functions for episodic memory.
    """

    def __init__(self):
        super().__init__(EpisodicMemory)
        self.vectorize_service = get_vectorize_service()

    # ==================== Basic CRUD Methods ====================

    async def get_by_event_id(
        self, event_id: str, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[EpisodicMemory]:
        """
        Retrieve episodic memory by event ID and user ID

        Args:
            event_id: Event ID
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            EpisodicMemory or None
        """
        try:
            # Convert string event_id to ObjectId
            object_id = ObjectId(event_id)
            result = await self.model.find_one(
                {"_id": object_id, "user_id": user_id}, session=session
            )
            if result:
                logger.debug(
                    "✅ Successfully retrieved episodic memory by event ID and user ID: %s",
                    event_id,
                )
            else:
                logger.debug(
                    "ℹ️  Episodic memory not found: event_id=%s, user_id=%s",
                    event_id,
                    user_id,
                )
            return result
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve episodic memory by event ID and user ID: %s", e
            )
            return None

    async def get_by_event_ids(
        self,
        event_ids: List[str],
        user_id: str,
        session: Optional[AsyncClientSession] = None,
    ) -> Dict[str, EpisodicMemory]:
        """
        Batch retrieve episodic memories by event ID list and user ID

        Args:
            event_ids: List of event IDs
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Dict[str, EpisodicMemory]: Dictionary with event_id as key, for fast lookup
        """
        if not event_ids:
            return {}

        try:
            # Convert list of string event_ids to list of ObjectIds
            object_ids = []
            for event_id in event_ids:
                try:
                    object_ids.append(ObjectId(event_id))
                except Exception as e:
                    logger.warning(f"⚠️  Invalid event_id: {event_id}, error: {e}")
                    continue

            if not object_ids:
                return {}

            # Batch query
            query = {"_id": {"$in": object_ids}}
            if user_id:
                query["user_id"] = user_id

            results = await self.model.find(query, session=session).to_list()

            # Convert to dictionary for easier subsequent use
            result_dict = {str(doc.id): doc for doc in results}

            logger.debug(
                "✅ Successfully batch retrieved episodic memories: user_id=%s, requested %d, found %d",
                user_id,
                len(event_ids),
                len(result_dict),
            )
            return result_dict
        except Exception as e:
            logger.error("❌ Failed to batch retrieve episodic memories: %s", e)
            return {}

    async def find_by_ids(
        self,
        ids: List[str],
        projection_model: Optional[Type[Any]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[Union[EpisodicMemory, EpisodicMemoryProjection]]:
        """
        Batch fetch episodic memories by their _id list.

        Accepts both ObjectId-like strings and raw string IDs (for legacy
        datasets where _id is not an ObjectId).

        Args:
            ids: List of document _id strings
            projection_model: Optional projection model to reduce data transfer
                (e.g. EpisodicMemoryProjection skips the vector field)
            session: Optional MongoDB session

        Returns:
            List of EpisodicMemory or projection model instances
        """
        query_filter = _build_id_filter(ids)
        if query_filter is None:
            return []
        try:
            if projection_model is not None:
                return await self.model.find(
                    query_filter, projection_model=projection_model, session=session
                ).to_list()
            return await self.model.find(query_filter, session=session).to_list()
        except Exception as e:
            logger.error("❌ Failed to find episodic memories by ids: %s", e)
            return []

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[EpisodicMemory, EpisodicMemoryProjection]]:
        """
        Retrieve list of episodic memories by filters (user_id, group_ids, and/or time range)

        Args:
            user_id: User ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values (records with user_id as None or "")
                - Other values: Exact match
            group_ids: List of Group IDs
                - None: Skip group filtering
                - []: Empty array, skip filtering
                - ["g1"]: Single element array, exact match
                - ["g1", "g2"]: Multiple elements, use $in operator
            start_time: Optional start time (inclusive)
            end_time: Optional end time (exclusive)
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order
            session: Optional MongoDB session, for transaction support
            model: Projection model type, default is EpisodicMemory (full version),
                   can pass EpisodicMemoryProjection to exclude vector field

        Returns:
            List of episodic memory objects of specified type
        """
        try:
            # Build query filter
            filter_dict = {}

            # Handle time range filter
            if start_time is not None and end_time is not None:
                filter_dict["timestamp"] = {"$gte": start_time, "$lt": end_time}
            elif start_time is not None:
                filter_dict["timestamp"] = {"$gte": start_time}
            elif end_time is not None:
                filter_dict["timestamp"] = {"$lt": end_time}

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    # Explicitly filter for null or empty string
                    filter_dict["user_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["user_id"] = user_id

            # Handle group_ids filter (array, no MAGIC_ALL)
            if group_ids is not None and len(group_ids) > 0:
                if len(group_ids) == 1:
                    # Single element: exact match
                    filter_dict["group_id"] = group_ids[0]
                else:
                    # Multiple elements: use $in operator
                    filter_dict["group_id"] = {"$in": group_ids}
            # group_ids is None or empty: skip group filtering

            # If model is not specified, use full version
            target_model = model if model is not None else self.model

            # Determine whether to use projection based on model type
            if target_model == self.model:
                query = self.model.find(filter_dict, session=session)
            else:
                query = self.model.find(
                    filter_dict, projection_model=target_model, session=session
                )

            sort_field = "-timestamp" if sort_desc else "timestamp"
            query = query.sort(sort_field)

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            logger.debug(
                "🔍 EpisodicMemory.find_by_filters query: %s, sort=%s, skip=%s, limit=%s",
                query.get_filter_query(),
                sort_field,
                skip,
                limit,
            )

            results = await query.to_list()
            logger.debug(
                "✅ Successfully retrieved episodic memories: user_id=%s, group_ids=%s, time_range=[%s, %s), found %d records",
                user_id,
                group_ids,
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve episodic memories: %s", e)
            return []

    async def count_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Count episodic memories by filters (without pagination)

        Args:
            user_id: User ID filter (same semantics as find_by_filters)
            group_ids: Group IDs filter (same semantics as find_by_filters)
            start_time: Optional start time (inclusive)
            end_time: Optional end time (exclusive)
            session: Optional MongoDB session

        Returns:
            Total count of matching records
        """
        try:
            # Build query filter (same as find_by_filters)
            filter_dict = {}

            # Handle time range filter
            if start_time is not None and end_time is not None:
                filter_dict["timestamp"] = {"$gte": start_time, "$lt": end_time}
            elif start_time is not None:
                filter_dict["timestamp"] = {"$gte": start_time}
            elif end_time is not None:
                filter_dict["timestamp"] = {"$lt": end_time}

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    filter_dict["user_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["user_id"] = user_id

            # Handle group_ids filter
            if group_ids is not None and len(group_ids) > 0:
                if len(group_ids) == 1:
                    filter_dict["group_id"] = group_ids[0]
                else:
                    filter_dict["group_id"] = {"$in": group_ids}

            count = await self.model.find(filter_dict, session=session).count()
            logger.debug(
                "✅ Counted episodic memories: user_id=%s, group_ids=%s, count=%d",
                user_id,
                group_ids,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count episodic memories: %s", e)
            return 0

    async def append_episodic_memory(
        self,
        episodic_memory: EpisodicMemory,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[EpisodicMemory]:
        """
        Append new episodic memory

        Args:
            episodic_memory: Episodic memory object
            session: Optional MongoDB session, for transaction support

        Returns:
            Appended EpisodicMemory or None
        """

        # Synchronize vector
        if episodic_memory.episode and not episodic_memory.vector:
            try:
                vector = await self.vectorize_service.get_embedding(
                    episodic_memory.episode
                )
                episodic_memory.vector = vector.tolist()
                # Set vectorization model information
                episodic_memory.vector_model = self.vectorize_service.get_model_name()
            except Exception as e:
                logger.error("❌ Failed to synchronize vector: %s", e)
        try:
            await episodic_memory.insert(session=session)
            logger.info(
                "✅ Successfully appended episodic memory: event_id=%s, user_id=%s",
                episodic_memory.event_id,
                episodic_memory.user_id,
            )
            return episodic_memory
        except Exception as e:
            logger.error("❌ Failed to append episodic memory: %s", e)
            return None

    async def delete_by_event_id(
        self, event_id: str, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete episodic memory by event ID and user ID

        Args:
            event_id: Event ID
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            # Convert string event_id to ObjectId
            object_id = ObjectId(event_id)
            # Directly delete and check deletion count
            result = await self.model.find(
                {"_id": object_id, "user_id": user_id}, session=session
            ).delete()

            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            success = deleted_count > 0

            if success:
                logger.info(
                    "✅ Successfully deleted episodic memory by event ID and user ID: %s",
                    event_id,
                )
                return True
            else:
                logger.warning(
                    "⚠️  Episodic memory to delete not found: event_id=%s, user_id=%s",
                    event_id,
                    user_id,
                )
                return False
        except Exception as e:
            logger.error(
                "❌ Failed to delete episodic memory by event ID and user ID: %s", e
            )
            return False

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Delete all episodic memories by user ID

        Args:
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of deleted records
        """
        try:
            result = await self.model.find({"user_id": user_id}).delete(session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully deleted episodic memories by user ID: %s, deleted %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to delete episodic memories by user ID: %s", e)
            return 0

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        parent_id: Optional[str] = None,
        session_id: Optional[str] = MAGIC_ALL,
        sender_id: Optional[str] = MAGIC_ALL,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete episodic memories by filter conditions

        Three-state filter semantics (user_id, group_id, session_id, sender_id):
        - MAGIC_ALL (default): skip this filter
        - None or "": match null/empty records
        - other value: exact match

        Args:
            user_id: User ID filter
            group_id: Group ID filter
            parent_id: Parent ID filter (for cascade delete)
            session_id: Session ID filter
            sender_id: Sender ID filter (maps to "sender_ids" field)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of deleted records
        """
        filter_dict: Dict[str, Any] = {}

        if user_id != MAGIC_ALL:
            if user_id is None or user_id == "":
                filter_dict["user_id"] = {"$in": [None, ""]}
            else:
                filter_dict["user_id"] = user_id

        if group_id != MAGIC_ALL:
            if group_id is None or group_id == "":
                filter_dict["group_id"] = {"$in": [None, ""]}
            else:
                filter_dict["group_id"] = group_id

        if parent_id is not None:
            filter_dict["parent_id"] = parent_id

        if session_id != MAGIC_ALL:
            if session_id is None or session_id == "":
                filter_dict["session_id"] = {"$in": [None, ""]}
            else:
                filter_dict["session_id"] = session_id

        if sender_id != MAGIC_ALL:
            if sender_id is None or sender_id == "":
                filter_dict["sender_ids"] = {"$in": [None, ""]}
            else:
                filter_dict["sender_ids"] = sender_id

        if not filter_dict:
            logger.warning("No filter conditions provided for delete_by_filters")
            return 0

        result = await self.model.delete_many(filter_dict, session=session)
        count = result.modified_count if result else 0
        logger.info(
            "Soft deleted episodic memories by filters: filter=%s, deleted %d records",
            filter_dict,
            count,
        )
        return count

    async def find_by_filter_paginated(
        self,
        query_filter: Optional[Dict[str, Any]] = None,
        skip: int = 0,
        limit: int = 100,
        sort_field: str = "created_at",
        sort_desc: bool = False,
    ) -> List[EpisodicMemory]:
        """
        Paginated query of EpisodicMemory by filter conditions, used for data synchronization scenarios

        Args:
            query_filter: Query filter conditions, query all if None
            skip: Number of results to skip
            limit: Limit number of returned results
            sort_field: Sort field, default is created_at
            sort_desc: Whether to sort in descending order, default False (ascending)

        Returns:
            List of EpisodicMemory
        """
        try:
            # Build query
            filter_dict = query_filter if query_filter else {}
            query = self.model.find(filter_dict)

            # Sort
            if sort_desc:
                query = query.sort(f"-{sort_field}")
            else:
                query = query.sort(sort_field)

            # Paginate
            query = query.skip(skip).limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully paginated query of EpisodicMemory: filter=%s, skip=%d, limit=%d, found %d records",
                filter_dict,
                skip,
                limit,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to paginate query of EpisodicMemory: %s", e)
            return []


# Export
__all__ = ["EpisodicMemoryRawRepository"]
