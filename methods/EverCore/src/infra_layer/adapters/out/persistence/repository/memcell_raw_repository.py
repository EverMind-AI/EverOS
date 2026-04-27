"""
MemCell Native CRUD Repository

Native data access layer for MemCell based on Beanie ODM, providing complete CRUD operations.
Does not depend on domain layer interfaces, directly operates on MemCell document models.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Type
from bson import ObjectId
from pydantic import BaseModel
from beanie.operators import And, GTE, LT, Eq, RegEx, Or
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL

from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)

logger = get_logger(__name__)


@repository("memcell_raw_repository", primary=True)
class MemCellRawRepository(BaseRepository[MemCell]):
    """
    MemCell Native CRUD Repository

    Provides direct database operations on MemCell documents, including:
    - Basic CRUD operations (inherited from BaseRepository)
    - Composite queries and filtering
    - Batch operations
    - Statistics and aggregation queries
    - Transaction management (inherited from BaseRepository)
    """

    def __init__(self):
        """Initialize repository"""
        super().__init__(MemCell)

    async def get_by_event_id(self, event_id: str) -> Optional[MemCell]:
        """
        Get MemCell by event_id

        Args:
            event_id: Event ID

        Returns:
            MemCell instance or None
        """
        try:
            result = await self.model.find_one({"_id": ObjectId(event_id)})
            if result:
                logger.debug(
                    "✅ Successfully retrieved MemCell by event_id: %s", event_id
                )
            else:
                logger.debug("⚠️  MemCell not found: event_id=%s", event_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve MemCell by event_id: %s", e)
            return None

    async def get_by_event_ids(
        self, event_ids: List[str], projection_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Batch get MemCell by event_id list

        Args:
            event_ids: List of event IDs
            projection_model: Pydantic projection model class, used to specify returned fields
                             For example: pass a Pydantic model containing only specific fields
                             None means return complete MemCell objects

        Returns:
            Dict[event_id, MemCell | ProjectionModel]: Mapping dictionary from event_id to MemCell (or projection model)
            Unfound event_ids will not appear in the dictionary
        """
        try:
            if not event_ids:
                logger.debug("⚠️  event_ids list is empty, returning empty dictionary")
                return {}

            # Convert event_id list to ObjectId list
            object_ids = []
            valid_event_ids = []  # Store valid original event_id strings
            for event_id in event_ids:
                try:
                    object_ids.append(ObjectId(event_id))
                    valid_event_ids.append(event_id)
                except Exception as e:
                    logger.warning("⚠️  Invalid event_id: %s, error: %s", event_id, e)

            if not object_ids:
                logger.debug("⚠️  No valid event_ids, returning empty dictionary")
                return {}

            # Build query
            query = self.model.find({"_id": {"$in": object_ids}})

            # Apply field projection
            # Use Beanie's .project() method, passing projection_model parameter
            if projection_model:
                query = query.project(projection_model=projection_model)

            # Batch query
            results = await query.to_list()

            # Create mapping dictionary from event_id to MemCell (or projection model)
            result_dict = {str(result.id): result for result in results}

            logger.debug(
                "✅ Successfully batch retrieved MemCell by event_ids: requested %d, found %d, projection: %s",
                len(event_ids),
                len(result_dict),
                "yes" if projection_model else "no",
            )

            return result_dict

        except Exception as e:
            logger.error("❌ Failed to batch retrieve MemCell by event_ids: %s", e)
            return {}

    async def append_memcell(
        self, memcell: MemCell, session: Optional[AsyncClientSession] = None
    ) -> Optional[MemCell]:
        """
        Append MemCell
        """
        try:
            await memcell.insert(session=session)
            logger.debug("Successfully appended MemCell: %s", memcell.event_id)
            return memcell
        except Exception as e:
            logger.error("❌ Failed to append MemCell: %s", e)
            return None

    async def update_by_event_id(
        self,
        event_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[MemCell]:
        """
        Update MemCell by event_id

        Args:
            event_id: Event ID
            update_data: Dictionary of update data
            session: Optional MongoDB session, for transaction support

        Returns:
            Updated MemCell instance or None
        """
        try:
            memcell = await self.get_by_event_id(event_id)
            if memcell:
                for key, value in update_data.items():
                    if hasattr(memcell, key):
                        setattr(memcell, key, value)
                await memcell.save(session=session)
                logger.debug(
                    "✅ Successfully updated MemCell by event_id: %s", event_id
                )
                return memcell
            return None
        except Exception as e:
            logger.error("❌ Failed to update MemCell by event_id: %s", e)
            raise e

    async def delete_by_event_id(
        self,
        event_id: str,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Soft delete MemCell by event_id

        Args:
            event_id: Event ID
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            memcell = await self.get_by_event_id(event_id)
            if memcell:
                await memcell.delete(deleted_by=deleted_by, session=session)
                logger.debug(
                    "✅ Successfully soft deleted MemCell by event_id: %s", event_id
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to soft delete MemCell by event_id: %s", e)
            return False

    async def hard_delete_by_event_id(
        self, event_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Hard delete (physical deletion) MemCell by event_id

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Args:
            event_id: Event ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            memcell = await self.model.hard_find_one({"_id": ObjectId(event_id)})
            if memcell:
                await memcell.hard_delete(session=session)
                logger.debug(
                    "✅ Successfully hard deleted MemCell by event_id: %s", event_id
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to hard delete MemCell by event_id: %s", e)
            return False

    # ==================== Query Methods ====================

    async def find_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        Query MemCell by user ID

        Args:
            user_id: User ID
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order

        Returns:
            List of MemCell
        """
        try:
            query = self.model.find({"user_id": user_id})

            # Sorting
            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            # Pagination
            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by user ID: %s, found %d records",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to query MemCell by user ID: %s", e)
            return []

    async def find_by_user_and_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        Query MemCell by user ID and time range

        Check both user_id field and participants array, match if user_id is in either

        Args:
            user_id: User ID
            start_time: Start time
            end_time: End time
            limit: Limit number of returned results
            skip: Number of results to skip

        Returns:
            List of MemCell
        """
        try:
            # Check both user_id field and participants array
            # Use OR logic: user_id matches OR user_id is in participants
            # Note: MongoDB automatically checks if array contains the value when using Eq on array fields
            query = self.model.find(
                And(
                    Or(
                        Eq(MemCell.user_id, user_id),
                        Eq(
                            MemCell.participants, user_id
                        ),  # MongoDB checks if array contains the value
                    ),
                    GTE(MemCell.timestamp, start_time),
                    LT(MemCell.timestamp, end_time),
                )
            ).sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by user and time range: %s, time range: %s - %s, found %d records",
                user_id,
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to query MemCell by user and time range: %s", e)
            return []

    async def find_by_group_id(
        self,
        group_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        Query MemCell by group ID

        Args:
            group_id: Group ID
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order

        Returns:
            List of MemCell
        """
        try:
            query = self.model.find({"group_id": group_id})

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by group ID: %s, found %d records",
                group_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to query MemCell by group ID: %s", e)
            return []

    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = False,
    ) -> List[MemCell]:
        """
        Query MemCell by time range

        Args:
            start_time: Start time
            end_time: End time
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order, default False (ascending)

        Returns:
            List of MemCell
        """
        try:
            query = self.model.find(
                {"timestamp": {"$gte": start_time, "$lt": end_time}}
            )

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by time range: time range: %s - %s, found %d records",
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to query MemCell by time range: %s", e)
            logger.error("Detailed error information:", exc_info=True)
            return []

    async def find_by_participants(
        self,
        participants: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        Query MemCell by participants

        Args:
            participants: List of participants
            match_all: Whether to match all participants (True) or any participant (False)
            limit: Limit number of returned results
            skip: Number of results to skip

        Returns:
            List of MemCell
        """
        try:
            if match_all:
                # Match all participants
                query = self.model.find({"participants": {"$all": participants}})
            else:
                # Match any participant
                query = self.model.find({"participants": {"$in": participants}})

            query = query.sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by participants: %s, match mode: %s, found %d records",
                participants,
                'all' if match_all else 'any',
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to query MemCell by participants: %s", e)
            return []

    async def search_by_keywords(
        self,
        keywords: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        Query MemCell by keywords

        Args:
            keywords: List of keywords
            match_all: Whether to match all keywords (True) or any keyword (False)
            limit: Limit number of returned results
            skip: Number of results to skip

        Returns:
            List of MemCell
        """
        try:
            if match_all:
                query = self.model.find({"keywords": {"$all": keywords}})
            else:
                query = self.model.find({"keywords": {"$in": keywords}})

            query = query.sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by keywords: %s, match mode: %s, found %d records",
                keywords,
                'all' if match_all else 'any',
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to query MemCell by keywords: %s", e)
            return []

    # ==================== Batch Operations ====================

    async def delete_by_user_id(
        self,
        user_id: str,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete all MemCell of a user

        Args:
            user_id: User ID
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of soft deleted records
        """
        try:
            result = await self.model.delete_many(
                {"user_id": user_id}, deleted_by=deleted_by, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully soft deleted all MemCell of user: %s, deleted %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to soft delete all MemCell of user: %s", e)
            return 0

    async def delete_by_group_id(
        self,
        group_id: str,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete all MemCell of a group

        Args:
            group_id: Group ID
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of soft deleted records
        """
        try:
            result = await self.model.delete_many(
                {"group_id": group_id}, deleted_by=deleted_by, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully soft deleted all MemCell of group: %s, deleted %d records",
                group_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to soft delete all MemCell of group: %s", e)
            return 0

    async def delete_by_filters(
        self,
        memcell_id: Optional[str] = None,
        group_id: Optional[str] = MAGIC_ALL,
        session_id: Optional[str] = MAGIC_ALL,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete MemCells by combined filter conditions

        Three-state filter semantics (group_id, session_id):
        - MAGIC_ALL (default): skip this filter
        - None or "": match null/empty records
        - other value: exact match

        Args:
            memcell_id: MemCell ID filter
            group_id: Group ID filter
            session_id: Session ID filter
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of soft deleted records
        """
        filter_dict = {}

        if memcell_id is not None:
            filter_dict["_id"] = ObjectId(memcell_id)

        if group_id != MAGIC_ALL:
            if group_id is None or group_id == "":
                filter_dict["group_id"] = {"$in": [None, ""]}
            else:
                filter_dict["group_id"] = group_id

        if session_id != MAGIC_ALL:
            if session_id is None or session_id == "":
                filter_dict["session_id"] = {"$in": [None, ""]}
            else:
                filter_dict["session_id"] = session_id

        if not filter_dict:
            logger.warning("No filter conditions provided for delete_by_filters")
            return 0

        result = await self.model.delete_many(
            filter_dict, deleted_by=deleted_by, session=session
        )
        count = result.modified_count if result else 0
        logger.info(
            "Soft deleted MemCells by filters: filter=%s, deleted %d records",
            filter_dict,
            count,
        )
        return count

    async def hard_delete_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Hard delete (physical deletion) all MemCell of a user

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Args:
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of hard deleted records
        """
        try:
            result = await self.model.hard_delete_many(
                {"user_id": user_id}, session=session
            )
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully hard deleted all MemCell of user: %s, deleted %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to hard delete all MemCell of user: %s", e)
            return 0

    async def delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete MemCell within time range

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of soft deleted records
        """
        try:
            filter_dict = {"timestamp": {"$gte": start_time, "$lt": end_time}}
            if user_id:
                filter_dict["user_id"] = user_id

            result = await self.model.delete_many(
                filter_dict, deleted_by=deleted_by, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully soft deleted MemCell within time range: %s - %s, user: %s, deleted %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to soft delete MemCell within time range: %s", e)
            return 0

    async def hard_delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Hard delete (physical deletion) MemCell within time range

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of hard deleted records
        """
        try:
            filter_dict = {"timestamp": {"$gte": start_time, "$lt": end_time}}
            if user_id:
                filter_dict["user_id"] = user_id

            result = await self.model.hard_delete_many(filter_dict, session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully hard deleted MemCell within time range: %s - %s, user: %s, deleted %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to hard delete MemCell within time range: %s", e)
            return 0

    # ==================== Soft Delete Recovery Methods ====================

    async def restore_by_event_id(
        self, event_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Restore soft-deleted MemCell by event_id

        Args:
            event_id: Event ID

        Returns:
            Returns True if restoration succeeds, otherwise False
        """
        try:
            memcell = await self.model.hard_find_one(
                {"_id": ObjectId(event_id)}, session=session
            )
            if memcell:
                await memcell.restore()
                logger.debug(
                    "✅ Successfully restored MemCell by event_id: %s", event_id
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to restore MemCell by event_id: %s", e)
            return False

    async def restore_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Restore all soft-deleted MemCell of a user

        Args:
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of restored records
        """
        try:
            result = await self.model.restore_many(
                {"user_id": user_id}, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully restored all MemCell of user: %s, restored %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to restore all MemCell of user: %s", e)
            return 0

    async def restore_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Restore soft-deleted MemCell within time range

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of restored records
        """
        try:
            filter_dict = {
                "timestamp": {"$gte": start_time, "$lt": end_time},
                "deleted_at": {"$ne": None},  # Only restore deleted records
            }
            if user_id:
                filter_dict["user_id"] = user_id

            result = await self.model.restore_many(filter_dict, session=session)
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully restored MemCell within time range: %s - %s, user: %s, restored %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to restore MemCell within time range: %s", e)
            return 0

    # ==================== Statistics and Aggregation Queries ====================

    async def count_by_user_id(self, user_id: str) -> int:
        """
        Count number of MemCell for a user

        Args:
            user_id: User ID

        Returns:
            Number of records
        """
        try:
            count = await self.model.find({"user_id": user_id}).count()
            logger.debug(
                "✅ Successfully counted user MemCell: %s, total %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count user MemCell: %s", e)
            return 0

    async def count_by_time_range(
        self, start_time: datetime, end_time: datetime, user_id: Optional[str] = None
    ) -> int:
        """
        Count number of MemCell within time range

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter

        Returns:
            Number of records
        """
        try:
            conditions = [
                GTE(MemCell.timestamp, start_time),
                LT(MemCell.timestamp, end_time),
            ]

            if user_id:
                conditions.append(Eq(MemCell.user_id, user_id))

            count = await self.model.find(And(*conditions)).count()
            logger.debug(
                "✅ Successfully counted MemCell within time range: %s - %s, user: %s, total %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count MemCell within time range: %s", e)
            return 0

    async def get_latest_by_user(self, user_id: str, limit: int = 10) -> List[MemCell]:
        """
        Get latest MemCell records for a user

        Args:
            user_id: User ID
            limit: Limit on number of returned records

        Returns:
            List of MemCell
        """
        try:
            results = (
                await self.model.find({"user_id": user_id})
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved latest user MemCell: %s, returned %d records",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve latest user MemCell: %s", e)
            return []


# Export
__all__ = ["MemCellRawRepository"]
