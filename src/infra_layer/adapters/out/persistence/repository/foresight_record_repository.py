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


@repository("foresight_record_repository", primary=True)
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
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve list of foresights by filters (user_id, group_id, and/or validity time range)

        Args:
            user_id: User ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values (records with user_id as None or "")
                - Other values: Exact match
            group_id: Group ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by group_id
                - None or "": Filter for null/empty values (records with group_id as None or "")
                - Other values: Exact match
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

            # Handle group_id filter
            if group_id != MAGIC_ALL:
                if group_id == "" or group_id is None:
                    # Explicitly filter for null or empty string
                    filter_dict["group_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["group_id"] = group_id

            # Use full version if model is not specified
            target_model = model if model is not None else self.model

            # Determine whether to use projection based on model type
            if target_model == self.model:
                query = self.model.find(filter_dict, session=session)
            else:
                query = self.model.find(
                    filter_dict, projection_model=target_model, session=session
                )

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Retrieved foresights successfully: user_id=%s, group_id=%s, time_range=[%s, %s), found %d records (model=%s)",
                user_id,
                group_id,
                start_str,
                end_str,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve foresights: %s", e)
            return []

    async def delete_expired(
        self,
        before: datetime,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Delete foresight records whose validity period ended before the given time.

        Args:
            before: Delete foresights with end_time strictly before this datetime.
                    end_time is stored as an ISO date string (YYYY-MM-DD).
            session: Optional MongoDB session for transaction support

        Returns:
            Number of deleted records
        """
        try:
            from common_utils.datetime_utils import to_date_str

            before_str = to_date_str(before)
            query_filter: Dict[str, Any] = {"end_time": {"$lt": before_str}}
            result = await self.model.find(query_filter, session=session).delete()
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Deleted expired foresights from MongoDB (before=%s): %d records",
                before_str,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to delete expired foresights from MongoDB: %s", e)
            return 0

    async def delete_by_id(
        self, memory_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete personal foresight by ID

        Args:
            memory_id: Memory ID
            session: Optional MongoDB session for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            object_id = ObjectId(memory_id)
            result = await self.model.find({"_id": object_id}, session=session).delete()
            success = result.deleted_count > 0 if result else False

            if success:
                logger.info("✅ Deleted personal foresight successfully: %s", memory_id)
            else:
                logger.warning(
                    "⚠️  Personal foresight to delete not found: %s", memory_id
                )

            return success
        except Exception as e:
            logger.error("❌ Failed to delete personal foresight: %s", e)
            return False

    async def delete_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Delete all foresights by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session for transaction support

        Returns:
            Number of deleted records
        """
        try:
            query_filter = {"parent_id": parent_id}
            if parent_type is not None:
                query_filter["parent_type"] = parent_type

            result = await self.model.find(query_filter, session=session).delete()
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Deleted foresights by parent memory ID successfully: %s (type=%s), deleted %d records",
                parent_id,
                parent_type,
                count,
            )
            return count
        except Exception as e:
            logger.error(
                "❌ Failed to delete foresights by parent episodic memory ID: %s", e
            )
            return 0


# Export
__all__ = ["ForesightRecordRawRepository"]
