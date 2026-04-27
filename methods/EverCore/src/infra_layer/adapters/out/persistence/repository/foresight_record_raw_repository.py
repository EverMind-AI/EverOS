"""
ForesightRecord Repository

Provides generic CRUD operations and query capabilities for foresight records.
"""

from datetime import datetime
from typing import List, Optional, Type, TypeVar, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from common_utils.datetime_utils import to_date_str
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
    ForesightRecordProjection,
)

# Define generic type variable
T = TypeVar('T', ForesightRecord, ForesightRecordProjection)

logger = get_logger(__name__)


@repository("foresight_record_raw_repository", primary=True)
class ForesightRecordRawRepository(BaseRepository[ForesightRecord]):
    """
    Raw repository for personal foresight data

    Provides CRUD operations and basic query functions for personal foresight records.
    Note: Vectors should be generated during extraction; this Repository is not responsible for vector generation.
    """

    def __init__(self):
        super().__init__(ForesightRecord)

    # ==================== Basic CRUD Methods ====================

    async def save(
        self, foresight: ForesightRecord, session: Optional[AsyncClientSession] = None
    ) -> Optional[ForesightRecord]:
        """
        Save personal foresight record

        Args:
            foresight: ForesightRecord object
            session: Optional MongoDB session for transaction support

        Returns:
            Saved ForesightRecord or None
        """
        try:
            await foresight.insert(session=session)
            logger.info(
                "✅ Saved personal foresight successfully: id=%s, user_id=%s, parent_type=%s, parent_id=%s",
                foresight.id,
                foresight.user_id,
                foresight.parent_type,
                foresight.parent_id,
            )
            return foresight
        except Exception as e:
            logger.error("❌ Failed to save personal foresight: %s", e)
            return None

    async def get_by_id(
        self,
        memory_id: str,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> Optional[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve personal foresight by ID

        Args:
            memory_id: Memory ID
            session: Optional MongoDB session for transaction support
            model: Type of model to return, defaults to ForesightRecord (full version)

        Returns:
            Foresight object of specified type or None
        """
        try:
            object_id = ObjectId(memory_id)

            # Use full version if model is not specified
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
                    "✅ Retrieved personal foresight by ID successfully: %s (model=%s)",
                    memory_id,
                    target_model.__name__,
                )
            else:
                logger.debug("ℹ️  Personal foresight not found: id=%s", memory_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve personal foresight by ID: %s", e)
            return None

    async def get_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve all foresights by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session for transaction support
            model: Type of model to return, defaults to ForesightRecord (full version)

        Returns:
            List of foresight objects of specified type
        """
        try:
            # Use full version if model is not specified
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
                "✅ Retrieved foresights by parent memory ID successfully: %s (type=%s), found %d records (model=%s)",
                parent_id,
                parent_type,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve foresights by parent episodic memory ID: %s", e
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
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve list of foresights by filters (user_id, group_ids, and/or validity time range)

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
            start_time: Optional query start time (datetime object)
                - Filters foresights whose validity period overlaps with [start_time, end_time)
                - Will be converted to ISO date string (YYYY-MM-DD) internally
            end_time: Optional query end time (datetime object)
                - Filters foresights whose validity period overlaps with [start_time, end_time)
                - Will be converted to ISO date string (YYYY-MM-DD) internally
            limit: Limit number of returned records
            skip: Number of records to skip
            session: Optional MongoDB session for transaction support
            model: Type of model to return, defaults to ForesightRecord (full version)

        Returns:
            List of foresight objects of specified type
        """
        try:
            # Build query filter
            filter_dict = {}

            # Convert datetime to ISO date string for foresight validity period comparison
            start_str = to_date_str(start_time)
            end_str = to_date_str(end_time)

            # Handle time range filter (overlap query)
            # Logic: foresight.start_time <= query.end_time AND foresight.end_time >= query.start_time
            if start_str is not None and end_str is not None:
                filter_dict["$and"] = [
                    {"start_time": {"$lte": end_str}},
                    {"end_time": {"$gte": start_str}},
                ]
            elif start_str is not None:
                # Only start_time: find foresights that end after start_time
                filter_dict["end_time"] = {"$gte": start_str}
            elif end_str is not None:
                # Only end_time: find foresights that start before end_time
                filter_dict["start_time"] = {"$lte": end_str}

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

            # Use full version if model is not specified
            target_model = model if model is not None else self.model

            # Determine whether to use projection based on model type
            if target_model == self.model:
                query = self.model.find(filter_dict, session=session)
            else:
                query = self.model.find(
                    filter_dict, projection_model=target_model, session=session
                )

            # Sort by created_at descending (most recent first)
            query = query.sort("-created_at")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            logger.debug(
                "🔍 ForesightRecord.find_by_filters query: %s, sort=-created_at, skip=%s, limit=%s",
                query.get_filter_query(),
                skip,
                limit,
            )

            results = await query.to_list()
            logger.debug(
                "✅ Retrieved foresights successfully: user_id=%s, group_ids=%s, time_range=[%s, %s), found %d records (model=%s)",
                user_id,
                group_ids,
                start_str,
                end_str,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve foresights: %s", e)
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
        Count foresights by filters (without pagination)

        Args:
            user_id: User ID filter (same semantics as find_by_filters)
            group_ids: Group IDs filter (same semantics as find_by_filters)
            start_time: Optional query start time (datetime object)
            end_time: Optional query end time (datetime object)
            session: Optional MongoDB session

        Returns:
            Total count of matching records
        """
        try:
            # Build query filter (same as find_by_filters)
            filter_dict = {}

            # Convert datetime to ISO date string for foresight validity period comparison
            start_str = to_date_str(start_time)
            end_str = to_date_str(end_time)

            # Handle time range filter (overlap query)
            if start_str is not None and end_str is not None:
                filter_dict["$and"] = [
                    {"start_time": {"$lte": end_str}},
                    {"end_time": {"$gte": start_str}},
                ]
            elif start_str is not None:
                filter_dict["end_time"] = {"$gte": start_str}
            elif end_str is not None:
                filter_dict["start_time"] = {"$lte": end_str}

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
                "✅ Counted foresights: user_id=%s, group_ids=%s, count=%d",
                user_id,
                group_ids,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count foresights: %s", e)
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
        Soft delete foresight records by filter conditions

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
            session: Optional MongoDB session for transaction support

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
            "Soft deleted foresight records by filters: filter=%s, deleted %d records",
            filter_dict,
            count,
        )
        return count


# Export
__all__ = ["ForesightRecordRawRepository"]
