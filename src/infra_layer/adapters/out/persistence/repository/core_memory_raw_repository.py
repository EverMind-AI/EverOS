from typing import List, Optional, Dict, Any, Tuple, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.core_memory import CoreMemory

logger = get_logger(__name__)


@repository("core_memory_raw_repository", primary=True)
class CoreMemoryRawRepository(BaseRepository[CoreMemory]):
    """
    Core memory raw data repository

    Provides CRUD operations and query functions for core memory.
    A single document contains data of two memory types: BaseMemory and Profile.
    (Preference-related fields have been merged into Profile)
    """

    def __init__(self):
        super().__init__(CoreMemory)

    # ==================== Version Management Methods ====================

    async def ensure_latest(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Ensure the latest version flag is correct for the specified user

        Find the latest version by user_id, set its is_latest to True, and set others to False.
        This is an idempotent operation and can be safely called repeatedly.

        Args:
            user_id: User ID
            session: Optional MongoDB session for transaction support

        Returns:
            Whether the update was successful
        """
        try:
            # Query only the most recent record (optimize performance)
            latest_version = await self.model.find_one(
                {"user_id": user_id}, sort=[("version", -1)], session=session
            )

            if not latest_version:
                logger.debug("ℹ️  No core memory found to update: user_id=%s", user_id)
                return True

            # Bulk update: set is_latest to False for all old versions
            await self.model.find(
                {"user_id": user_id, "version": {"$ne": latest_version.version}},
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # Update the latest version's is_latest to True
            if latest_version.is_latest is not True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ Set latest version flag: user_id=%s, version=%s",
                    user_id,
                    latest_version.version,
                )

            return True
        except Exception as e:
            logger.error(
                "❌ Failed to ensure latest version flag: user_id=%s, error=%s",
                user_id,
                e,
            )
            return False

    # ==================== Basic CRUD Methods ====================

    async def get_by_user_id(
        self,
        user_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Union[Optional[CoreMemory], List[CoreMemory]]:
        """
        Get core memory by user ID

        Args:
            user_id: User ID
            version_range: Version range (start, end), inclusive interval [start, end].
                          If not provided or None, get the latest version (sorted by version descending)
                          If provided, return all versions within the range
            session: Optional MongoDB session for transaction support

        Returns:
            If version_range is None, return a single CoreMemory or None
            If version_range is not None, return List[CoreMemory]
        """
        try:
            query_filter = {"user_id": user_id}

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

            # If no version range is specified, get the latest version (single result)
            if version_range is None:
                result = await self.model.find_one(
                    query_filter,
                    sort=[
                        ("version", -1)
                    ],  # Sort by version descending to get the latest
                    session=session,
                )
                if result:
                    logger.debug(
                        "✅ Successfully retrieved core memory by user ID: %s, version=%s",
                        user_id,
                        result.version,
                    )
                else:
                    logger.debug("ℹ️  Core memory not found: user_id=%s", user_id)
                return result
            else:
                # If version range is specified, get all matching versions
                results = await self.model.find(
                    query_filter,
                    sort=[("version", -1)],
                    session=session,  # Sort by version descending
                ).to_list()
                logger.debug(
                    "✅ Successfully retrieved core memory versions by user ID: %s, version_range=%s, found %d records",
                    user_id,
                    version_range,
                    len(results),
                )
                return results
        except Exception as e:
            logger.error("❌ Failed to retrieve core memory by user ID: %s", e)
            return None if version_range is None else []

    async def update_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[CoreMemory]:
        """
        Update core memory by user ID

        Args:
            user_id: User ID
            update_data: Update data
            version: Optional version number; if specified, update the specific version, otherwise update the latest version
            session: Optional MongoDB session for transaction support

        Returns:
            Updated CoreMemory or None
        """
        try:
            # Find the document to update
            if version is not None:
                # Update specific version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "version": version}, session=session
                )
            else:
                # Update latest version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id}, sort=[("version", -1)], session=session
                )

            if not existing_doc:
                logger.warning(
                    "⚠️  Core memory not found for update: user_id=%s, version=%s",
                    user_id,
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
                "✅ Successfully updated core memory by user ID: user_id=%s, version=%s",
                user_id,
                existing_doc.version,
            )

            return existing_doc
        except Exception as e:
            logger.error("❌ Failed to update core memory by user ID: %s", e)
            return None

    async def delete_by_user_id(
        self,
        user_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Delete core memory by user ID

        Args:
            user_id: User ID
            version: Optional version number; if specified, delete only that version, otherwise delete all versions
            session: Optional MongoDB session for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            query_filter = {"user_id": user_id}
            if version is not None:
                query_filter["version"] = version

            if version is not None:
                # Delete specific version - delete directly and check deletion count
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ Successfully deleted core memory by user ID and version: user_id=%s, version=%s",
                        user_id,
                        version,
                    )
                    # After deletion, ensure the latest version flag is correct
                    await self.ensure_latest(user_id, session)
                else:
                    logger.warning(
                        "⚠️  Core memory not found for deletion: user_id=%s, version=%s",
                        user_id,
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
                        "✅ Successfully deleted all core memory by user ID: user_id=%s, deleted %d records",
                        user_id,
                        deleted_count,
                    )
                else:
                    logger.warning(
                        "⚠️  Core memory not found for deletion: user_id=%s", user_id
                    )

            return success
        except Exception as e:
            logger.error("❌ Failed to delete core memory by user ID: %s", e)
            return False

    async def upsert_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[CoreMemory]:
        """
        Update or insert core memory by user ID

        If update_data contains a version field:
        - If that version exists, update it
        - If that version does not exist, create a new version (version must be provided)
        If update_data does not contain a version field:
        - Get the latest version and update it; if it doesn't exist, raise an error (version must be provided when creating)

        Args:
            user_id: User ID
            update_data: Data to update (must contain version field when creating a new version)
            session: Optional MongoDB session for transaction support

        Returns:
            Updated or created core memory record
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # If version is specified, find the specific version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "version": version}, session=session
                )
            else:
                # If version is not specified, find the latest version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id}, sort=[("version", -1)], session=session
                )

            if existing_doc:
                # Update existing record
                for key, value in update_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ Successfully updated existing core memory: user_id=%s, version=%s",
                    user_id,
                    existing_doc.version,
                )

                # If version was updated, ensure latest flag is correct
                if version is not None:
                    await self.ensure_latest(user_id, session)

                return existing_doc
            else:
                # When creating a new record, version must be provided
                if version is None:
                    logger.error(
                        "❌ Version field must be provided when creating new core memory: user_id=%s",
                        user_id,
                    )
                    raise ValueError(
                        f"Version field must be provided when creating new core memory: user_id={user_id}"
                    )

                # Create new record
                new_doc = CoreMemory(user_id=user_id, **update_data)
                await new_doc.create(session=session)
                logger.info(
                    "✅ Successfully created new core memory: user_id=%s, version=%s",
                    user_id,
                    new_doc.version,
                )

                # After creation, ensure latest version flag is correct
                await self.ensure_latest(user_id, session)

                return new_doc
        except ValueError:
            # Re-raise ValueError, do not catch it in Exception
            raise
        except Exception as e:
            logger.error("❌ Failed to update or create core memory: %s", e)
            return None

    # ==================== Field Extraction Methods ====================

    def get_base(self, memory: CoreMemory) -> Dict[str, Any]:
        """
        Get basic information

        Args:
            memory: CoreMemory instance

        Returns:
            Dictionary of basic information
        """
        return {
            "user_name": memory.user_name,
            "gender": memory.gender,
            "position": memory.position,
            "supervisor_user_id": memory.supervisor_user_id,
            "team_members": memory.team_members,
            "okr": memory.okr,
            "base_location": memory.base_location,
            "hiredate": memory.hiredate,
            "age": memory.age,
            "department": memory.department,
        }

    def get_profile(self, memory: CoreMemory) -> Dict[str, Any]:
        """
        Get personal profile

        Args:
            memory: CoreMemory instance

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
            "interests": getattr(memory, "interests", None),
            "tendency": memory.tendency,
        }

    async def find_by_user_ids(
        self,
        user_ids: List[str],
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[CoreMemory]:
        """
        Batch retrieve core memory by list of user IDs

        Args:
            user_ids: List of user IDs
            only_latest: Whether to retrieve only the latest version, default is True. Use is_latest field to filter latest versions in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of CoreMemory
        """
        try:
            if not user_ids:
                return []

            query_filter = {"user_id": {"$in": user_ids}}

            # In batch queries, use is_latest field to filter latest versions
            if only_latest:
                query_filter["is_latest"] = True

            query = self.model.find(query_filter, session=session)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully retrieved core memory by user ID list: %d user IDs, only_latest=%s, found %d records",
                len(user_ids),
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve core memory by user ID list: %s", e)
            return []


# Export
__all__ = ["CoreMemoryRawRepository"]
