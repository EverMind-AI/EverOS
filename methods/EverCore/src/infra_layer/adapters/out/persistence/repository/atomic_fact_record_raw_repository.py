"""
AtomicFactRecord Repository

Provides CRUD operations and query capabilities for generic atomic facts.
"""

from datetime import datetime
from typing import Any, List, Optional, Type, TypeVar, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.mongo.mongo_utils import build_id_filter as _build_id_filter
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.memory.atomic_fact_record import (
    AtomicFactRecord,
    AtomicFactRecordProjection,
)

# Define generic type variable
T = TypeVar('T', AtomicFactRecord, AtomicFactRecordProjection)

logger = get_logger(__name__)


@repository("atomic_fact_record_raw_repository", primary=True)
class AtomicFactRecordRawRepository(BaseRepository[AtomicFactRecord]):
    """
    Personal atomic fact raw data repository

    Provides CRUD operations and basic query functions for personal atomic facts.
    Note: Vectors should be generated during extraction; this Repository is not responsible for vector generation.
    """

    def __init__(self):
        super().__init__(AtomicFactRecord)

    # ==================== Basic CRUD Methods ====================

    async def save(
        self, record: AtomicFactRecord, session: Optional[AsyncClientSession] = None
    ) -> Optional[AtomicFactRecord]:
        """
        Save atomic fact record

        Args:
            record: AtomicFactRecord object
            session: Optional MongoDB session, for transaction support

        Returns:
            Saved AtomicFactRecord or None
        """
        try:
            await record.insert(session=session)
            logger.info(
                "Saved atomic fact record successfully: id=%s, user_id=%s, parent_type=%s, parent_id=%s",
                record.id,
                record.user_id,
                record.parent_type,
                record.parent_id,
            )
            return record
        except Exception as e:
            logger.error("Failed to save atomic fact record: %s", e)
            return None

    async def get_by_id(
        self,
        log_id: str,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> Optional[Union[AtomicFactRecord, AtomicFactRecordProjection]]:
        """
        Get personal atomic fact by ID

        Args:
            log_id: Log ID
            session: Optional MongoDB session, for transaction support
            model: Returned model type, default is AtomicFactRecord (full version), can pass AtomicFactRecordShort

        Returns:
            Atomic fact object of specified type or None
        """
        try:
            object_id = ObjectId(log_id)

            # If model is not specified, use full version
            target_model = model if model is not None else self.model

            # Determine whether to use projection based on model type
            if target_model == self.model:
                result = await self.model.find_one({"_id": object_id}, session=session)
            else:
                result = await self.model.find_one(
                    {"_id": object_id}, projection_model=target_model, session=session
                )

            if result:
                logger.debug(
                    "✅ Retrieved personal atomic fact by ID successfully: %s (model=%s)",
                    log_id,
                    target_model.__name__,
                )
            else:
                logger.debug("ℹ️  Personal atomic fact not found: id=%s", log_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve personal atomic fact by ID: %s", e)
            return None

    async def find_by_ids(
        self,
        ids: List[str],
        projection_model: Optional[Type[Any]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[Union[AtomicFactRecord, AtomicFactRecordProjection]]:
        """
        Batch fetch atomic facts by their _id list.

        Accepts both ObjectId-like strings and raw string IDs.

        Args:
            ids: List of document _id strings
            projection_model: Optional projection model to reduce data transfer
                (e.g. AtomicFactRecordProjection skips the vector field)
            session: Optional MongoDB session

        Returns:
            List of AtomicFactRecord or projection model instances
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
            logger.error("❌ Failed to find atomic facts by ids: %s", e)
            return []

    async def get_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[AtomicFactRecord, AtomicFactRecordProjection]]:
        """
        Get all atomic facts by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session, for transaction support
            model: Returned model type, default is AtomicFactRecord (full version), can pass AtomicFactRecordShort

        Returns:
            List of atomic fact objects of specified type
        """
        try:
            # If model is not specified, use full version
            target_model = model if model is not None else self.model

            # Build query filter
            query_filter = {"parent_id": parent_id}
            if parent_type:
                query_filter["parent_type"] = parent_type

            # Determine whether to use projection based on model type
            if target_model == self.model:
                query = self.model.find(query_filter, session=session)
            else:
                query = self.model.find(
                    query_filter, projection_model=target_model, session=session
                )

            results = await query.to_list()
            logger.debug(
                "✅ Retrieved atomic facts by parent memory ID successfully: %s (type=%s), found %d records (model=%s)",
                parent_id,
                parent_type,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve atomic facts by parent episodic memory ID: %s", e
            )
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
    ) -> List[Union[AtomicFactRecord, AtomicFactRecordProjection]]:
        """
        Get list of atomic facts by filters (user_id, group_ids, and/or time range)

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
            limit: Limit number of returned records
            skip: Number of records to skip
            sort_desc: Whether to sort by time in descending order
            session: Optional MongoDB session, for transaction support
            model: Returned model type, default is AtomicFactRecord (full version), can pass AtomicFactRecordProjection

        Returns:
            List of atomic fact objects of specified type
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
                "🔍 AtomicFactRecord.find_by_filters query: %s, sort=%s, skip=%s, limit=%s",
                query.get_filter_query(),
                sort_field,
                skip,
                limit,
            )

            results = await query.to_list()
            logger.debug(
                "✅ Retrieved atomic facts successfully: user_id=%s, group_ids=%s, time_range=[%s, %s), found %d records (model=%s)",
                user_id,
                group_ids,
                start_time,
                end_time,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve atomic facts: %s", e)
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
        Count atomic facts by filters (without pagination)

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
                "✅ Counted atomic facts: user_id=%s, group_ids=%s, count=%d",
                user_id,
                group_ids,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count atomic facts: %s", e)
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
        Soft delete atomic fact records by filter conditions

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
        filter_dict = {}

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
            "Soft deleted atomic fact records by filters: filter=%s, deleted %d records",
            filter_dict,
            count,
        )
        return count


# Export
__all__ = ["AtomicFactRecordRawRepository"]
