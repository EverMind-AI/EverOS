"""
Native CRUD repository for GroupUserProfileMemory

Native data access layer for GroupUserProfileMemory based on Beanie ODM, providing complete CRUD operations.
Does not depend on domain layer interfaces, directly operates on GroupUserProfileMemory document models.
Supports joint queries and operations based on user_id and group_id.
"""

from typing import List, Optional, Dict, Any, Tuple
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.group_user_profile_memory import (
    GroupUserProfileMemory,
)

logger = get_logger(__name__)


@repository("group_user_profile_memory_raw_repository", primary=True)
class GroupUserProfileMemoryRawRepository(BaseRepository[GroupUserProfileMemory]):
    """
    Native CRUD repository for GroupUserProfileMemory

    Provides direct database operations on GroupUserProfileMemory documents, including:
    - Basic CRUD operations (inherited from BaseRepository)
    - Joint queries based on user_id and group_id
    - Individual queries based on user_id or group_id
    - Batch queries and operations
    - Specialized methods related to profiles
    - Transaction management (inherited from BaseRepository)
    """

    def __init__(self):
        """Initialize the repository"""
        super().__init__(GroupUserProfileMemory)

    # ==================== Version Management Methods ====================

    async def ensure_latest(
        self, user_id: str, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Ensure the latest version flag is correctly set for the specified user in the specified group

        Find the latest version by user_id and group_id, set its is_latest to True, and set others to False.
        This is an idempotent operation and can be safely called repeatedly.

        Args:
            user_id: User ID
            group_id: Group ID
            session: Optional MongoDB session for transaction support

        Returns:
            Whether the update was successful
        """
        try:
            # Only query the most recent record (optimize performance)
            latest_version = await self.model.find_one(
                {"user_id": user_id, "group_id": group_id},
                sort=[("version", -1)],
                session=session,
            )

            if not latest_version:
                logger.debug(
                    "ℹ️  No group user profile found to update: user_id=%s, group_id=%s",
                    user_id,
                    group_id,
                )
                return True

            # Batch update: set is_latest to False for all older versions
            await self.model.find(
                {
                    "user_id": user_id,
                    "group_id": group_id,
                    "version": {"$ne": latest_version.version},
                },
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # Update the latest version's is_latest to True
            if latest_version.is_latest is not True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ Set latest version flag: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    latest_version.version,
                )

            return True
        except Exception as e:
            logger.error(
                "❌ Failed to ensure latest version flag: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            return False

    # ==================== CRUD Methods Based on Composite Key ====================

    async def get_by_user_group(
        self,
        user_id: str,
        group_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GroupUserProfileMemory]:
        """
        Get group user profile memory by user ID and group ID

        Args:
            user_id: User ID
            group_id: Group ID
            version_range: Version range (start, end), closed interval [start, end].
                          If not provided or None, get the latest version (sorted by version descending)
            session: Optional MongoDB session for transaction support

        Returns:
            GroupUserProfileMemory instance or None
        """
        try:
            query_filter = {"user_id": user_id, "group_id": group_id}

            # Handle version range query
            if version_range:
                start_version, end_version = version_range
                version_filter = {}
                if start_version is not None:
                    version_filter["$gte"] = start_version
                if end_version is not None:
                    version_filter["$lte"] = end_version
                if version_filter:
                    query_filter["version"] = version_filter

            # Sort by version descending, get the latest version
            result = await self.model.find_one(
                query_filter, sort=[("version", -1)], session=session
            )

            if result:
                logger.debug(
                    "✅ Successfully retrieved group user profile by user ID and group ID: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    result.version,
                )
            else:
                logger.debug(
                    "ℹ️ Group user profile not found: user_id=%s, group_id=%s",
                    user_id,
                    group_id,
                )
            return result
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve group user profile by user ID and group ID: %s",
                e,
            )
            return None

    async def batch_get_by_user_groups(
        self,
        user_group_pairs: List[Tuple[str, str]],
        session: Optional[AsyncClientSession] = None,
    ) -> Dict[Tuple[str, str], Optional[GroupUserProfileMemory]]:
        """
        Batch retrieve group user profile memory by user ID and group ID

        Args:
            user_group_pairs: List of (user_id, group_id) tuples
            session: Optional MongoDB session for transaction support

        Returns:
            Dict[(user_id, group_id), GroupUserProfileMemory]: Mapping dictionary
        """
        try:
            if not user_group_pairs:
                return {}

            # Deduplicate
            unique_pairs = list(set(user_group_pairs))
            logger.debug(
                "Batch retrieving group user profiles: total %d (before deduplication: %d)",
                len(unique_pairs),
                len(user_group_pairs),
            )

            # Construct query conditions: retrieve the latest version for all (user_id, group_id) pairs
            # Use aggregation pipeline to achieve batch query for latest versions
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"user_id": user_id, "group_id": group_id}
                            for user_id, group_id in unique_pairs
                        ]
                    }
                },
                # Group by user_id, group_id, version, get the latest version for each group
                {"$sort": {"user_id": 1, "group_id": 1, "version": -1}},
                {
                    "$group": {
                        "_id": {"user_id": "$user_id", "group_id": "$group_id"},
                        "doc": {"$first": "$$ROOT"},
                    }
                },
                {"$replaceRoot": {"newRoot": "$doc"}},
            ]

            # Execute aggregation query
            collection = self.model.get_pymongo_collection()
            cursor = await collection.aggregate(pipeline, session=session)
            results = await cursor.to_list(length=None)

            # Build result dictionary
            result_dict = {}
            for doc in results:
                if not doc:
                    continue
                memory = GroupUserProfileMemory.model_validate(doc)
                key = (memory.user_id, memory.group_id)
                result_dict[key] = memory

            # Fill missing records with None
            for pair in unique_pairs:
                if pair not in result_dict:
                    result_dict[pair] = None

            logger.debug(
                "✅ Batch retrieval of group user profiles completed: successfully retrieved %d",
                len([v for v in result_dict.values() if v is not None]),
            )

            return result_dict
        except Exception as e:
            logger.error("❌ Failed to batch retrieve group user profiles: %s", e)
            return {}

    async def update_by_user_group(
        self,
        user_id: str,
        group_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GroupUserProfileMemory]:
        """
        Update group user profile memory by user ID and group ID

        Args:
            user_id: User ID
            group_id: Group ID
            update_data: Update data
            version: Optional version number; if specified, update the specific version, otherwise update the latest version
            session: Optional MongoDB session for transaction support

        Returns:
            Updated GroupUserProfileMemory or None
        """
        try:
            # Find the document to update
            if version is not None:
                # Update specific version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id, "version": version},
                    session=session,
                )
            else:
                # Update latest version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id},
                    sort=[("version", -1)],
                    session=session,
                )

            if not existing_doc:
                logger.warning(
                    "⚠️ Group user profile to update not found: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    version,
                )
                return None

            # Update document
            for key, value in update_data.items():
                if hasattr(existing_doc, key):
                    setattr(existing_doc, key, value)

            # Save updated document
            await existing_doc.save(session=session)
            logger.debug(
                "✅ Successfully updated group user profile by user ID and group ID: user_id=%s, group_id=%s, version=%s",
                user_id,
                group_id,
                existing_doc.version,
            )

            return existing_doc
        except Exception as e:
            logger.error(
                "❌ Failed to update group user profile by user ID and group ID: %s", e
            )
            return None

    async def delete_by_user_group(
        self,
        user_id: str,
        group_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Delete group user profile memory by user ID and group ID

        Args:
            user_id: User ID
            group_id: Group ID
            version: Optional version number; if specified, only delete the specific version, otherwise delete all versions
            session: Optional MongoDB session for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            query_filter = {"user_id": user_id, "group_id": group_id}
            if version is not None:
                query_filter["version"] = version

            if version is not None:
                # Delete specific version - directly delete and check deletion count
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ Successfully deleted group user profile by user ID, group ID, and version: user_id=%s, group_id=%s, version=%s",
                        user_id,
                        group_id,
                        version,
                    )
                    # After deletion, ensure the latest version flag is correct
                    await self.ensure_latest(user_id, group_id, session)
                else:
                    logger.warning(
                        "⚠️ Group user profile to delete not found: user_id=%s, group_id=%s, version=%s",
                        user_id,
                        group_id,
                        version,
                    )
            else:
                # Delete all versions
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ Successfully deleted all group user profiles by user ID and group ID: user_id=%s, group_id=%s, deleted %d records",
                        user_id,
                        group_id,
                        deleted_count,
                    )
                else:
                    logger.warning(
                        "⚠️ Group user profile to delete not found: user_id=%s, group_id=%s",
                        user_id,
                        group_id,
                    )

            return success
        except Exception as e:
            logger.error(
                "❌ Failed to delete group user profile by user ID and group ID: %s", e
            )
            return False

    async def upsert_by_user_group(
        self,
        user_id: str,
        group_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GroupUserProfileMemory]:
        """
        Update or insert group user profile memory by user ID and group ID

        If the update_data contains a version field:
        - If that version exists, update it
        - If that version does not exist, create a new version (version must be provided)
        If update_data does not contain a version field:
        - Get the latest version and update it; if it does not exist, raise an error (version must be provided when creating)

        Args:
            user_id: User ID
            group_id: Group ID
            update_data: Data to update (must contain version field when creating a new version)
            session: Optional MongoDB session for transaction support

        Returns:
            Updated or created group user profile record
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # If version is specified, find the specific version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id, "version": version},
                    session=session,
                )
            else:
                # If version is not specified, find the latest version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id},
                    sort=[("version", -1)],
                    session=session,
                )

            if existing_doc:
                # Update existing record
                for key, value in update_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ Successfully updated existing group user profile: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    existing_doc.version,
                )

                # If version was updated, ensure the latest flag is correct
                if version is not None:
                    await self.ensure_latest(user_id, group_id, session)

                return existing_doc
            else:
                # When creating a new record, version must be provided
                if version is None:
                    logger.error(
                        "❌ Version field must be provided when creating a new group user profile: user_id=%s, group_id=%s",
                        user_id,
                        group_id,
                    )
                    raise ValueError(
                        f"Version field must be provided when creating a new group user profile: user_id={user_id}, group_id={group_id}"
                    )

                # Create new record
                new_doc = GroupUserProfileMemory(
                    user_id=user_id, group_id=group_id, **update_data
                )
                await new_doc.create(session=session)
                logger.info(
                    "✅ Successfully created new group user profile: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    new_doc.version,
                )

                # After creation, ensure the latest version flag is correct
                await self.ensure_latest(user_id, group_id, session)

                return new_doc
        except ValueError:
            # Re-raise ValueError, do not catch it in Exception
            raise
        except Exception as e:
            logger.error("❌ Failed to update or create group user profile: %s", e)
            return None

    # ==================== Single-Key Query Methods ====================

    async def get_by_user_id(
        self,
        user_id: str,
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        Get profile memories of the user in all groups by user ID

        Args:
            user_id: User ID
            only_latest: Whether to get only the latest version, default is True. Use is_latest field to filter latest version in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of GroupUserProfileMemory
        """
        try:
            query_filter = {"user_id": user_id}

            # In batch queries, use is_latest field to filter latest version
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()
            logger.debug(
                "✅ Successfully retrieved group user profiles by user ID: user_id=%s, only_latest=%s, found %d records",
                user_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve group user profiles by user ID: %s", e)
            return []

    async def get_by_group_id(
        self,
        group_id: str,
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        Get profile memories of all users in the group by group ID

        Args:
            group_id: Group ID
            only_latest: Whether to get only the latest version, default is True. Use is_latest field to filter latest version in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of GroupUserProfileMemory
        """
        try:
            query_filter = {"group_id": group_id}

            # In batch queries, use is_latest field to filter latest version
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()
            logger.debug(
                "✅ Successfully retrieved group user profiles by group ID: group_id=%s, only_latest=%s, found %d records",
                group_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve group user profiles by group ID: %s", e)
            return []

    # ==================== Batch Query Methods ====================

    async def get_by_user_ids(
        self,
        user_ids: List[str],
        group_id: Optional[str] = None,
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        Batch retrieve group user profile memories by list of user IDs

        Args:
            user_ids: List of user IDs
            group_id: Optional group ID; if provided, only query user profiles in that group
            only_latest: Whether to get only the latest version, default is True. Use is_latest field to filter latest version in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of GroupUserProfileMemory
        """
        try:
            if not user_ids:
                return []

            # Build query filter
            query_filter = {"user_id": {"$in": user_ids}}
            if group_id:
                query_filter["group_id"] = group_id

            # In batch queries, use is_latest field to filter latest version
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()

            logger.debug(
                "✅ Successfully retrieved group user profiles by user ID list: %d user IDs, group_id=%s, only_latest=%s, found %d records",
                len(user_ids),
                group_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve group user profiles by user ID list: %s", e
            )
            return []

    async def get_by_group_ids(
        self,
        group_ids: List[str],
        user_id: Optional[str] = None,
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        Batch retrieve group user profile memories by list of group IDs

        Args:
            group_ids: List of group IDs
            user_id: Optional user ID; if provided, only query the user's profiles in these groups
            only_latest: Whether to get only the latest version, default is True. Use is_latest field to filter latest version in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of GroupUserProfileMemory
        """
        try:
            if not group_ids:
                return []

            # Build query filter
            query_filter = {"group_id": {"$in": group_ids}}
            if user_id:
                query_filter["user_id"] = user_id

            # In batch queries, use is_latest field to filter latest version
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()

            logger.debug(
                "✅ Successfully retrieved group user profiles by group ID list: %d group IDs, user_id=%s, only_latest=%s, found %d records",
                len(group_ids),
                user_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve group user profiles by group ID list: %s", e
            )
            return []

    # ==================== Profile-Specific Methods ====================

    def get_profile(self, memory: GroupUserProfileMemory) -> Dict[str, Any]:
        """
        Get personal profile

        Args:
            memory: GroupUserProfileMemory instance

        Returns:
            Dictionary of personal profile
        """
        return {
            "hard_skills": memory.hard_skills,
            "soft_skills": memory.soft_skills,
            "personality": memory.personality,
            "projects_participated": memory.projects_participated,
            "user_goal": memory.user_goal,
            "work_responsibility": memory.work_responsibility,
            "working_habit_preference": memory.working_habit_preference,
            "interests": memory.interests,
            "tendency": memory.tendency,
        }

    # ==================== Deletion Methods ====================

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Delete user's profile memories in all groups

        Args:
            user_id: User ID
            session: Optional MongoDB session for transaction support

        Returns:
            Number of deleted records
        """
        try:
            result = await self.model.find(
                {"user_id": user_id}, session=session
            ).delete()
            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            logger.debug(
                "✅ Successfully deleted group user profiles by user ID: user_id=%s, deleted %d records",
                user_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error("❌ Failed to delete group user profiles by user ID: %s", e)
            return 0

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Delete profile memories of all users in the group

        Args:
            group_id: Group ID
            session: Optional MongoDB session for transaction support

        Returns:
            Number of deleted records
        """
        try:
            result = await self.model.find(
                {"group_id": group_id}, session=session
            ).delete()
            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            logger.debug(
                "✅ Successfully deleted group user profiles by group ID: group_id=%s, deleted %d records",
                group_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error("❌ Failed to delete group user profiles by group ID: %s", e)
            return 0


# Export
__all__ = ["GroupUserProfileMemoryRawRepository"]
