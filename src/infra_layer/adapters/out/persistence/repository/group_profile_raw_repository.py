from typing import List, Optional, Dict, Any, Tuple
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.group_profile import (
    GroupProfile,
)

logger = get_logger(__name__)


@repository("group_profile_raw_repository", primary=True)
class GroupProfileRawRepository(BaseRepository[GroupProfile]):
    """
    Group profile raw data repository

    Provides CRUD operations and query capabilities for group profiles.
    Supports management of group information, role definitions, user tags, and recent topics.
    """

    def __init__(self):
        super().__init__(GroupProfile)

    # ==================== Version Management Methods ====================

    async def ensure_latest(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Ensure the latest version flag is correctly set for the specified group

        Find the latest version by group_id, set its is_latest to True, and set others to False.
        This is an idempotent operation and can be safely called repeatedly.

        Args:
            group_id: Group ID
            session: Optional MongoDB session for transaction support

        Returns:
            Whether the update was successful
        """
        try:
            # Query only the most recent record (optimize performance)
            latest_version = await self.model.find_one(
                {"group_id": group_id}, sort=[("version", -1)], session=session
            )

            if not latest_version:
                logger.debug(
                    "ℹ️  No group profile found to update: group_id=%s", group_id
                )
                return True

            # Batch update: set is_latest to False for all older versions
            await self.model.find(
                {"group_id": group_id, "version": {"$ne": latest_version.version}},
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # Update the latest version's is_latest to True
            if latest_version.is_latest is not True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ Set latest version flag: group_id=%s, version=%s",
                    group_id,
                    latest_version.version,
                )

            return True
        except Exception as e:
            logger.error(
                "❌ Failed to ensure latest version flag: group_id=%s, error=%s",
                group_id,
                e,
            )
            return False

    # ==================== Basic CRUD Methods ====================

    async def get_by_group_id(
        self,
        group_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GroupProfile]:
        """
        Get group profile by group ID

        Args:
            group_id: Group ID
            version_range: Version range (start, end), inclusive [start, end].
                          If not provided or None, get the latest version (sorted by version descending)
            session: Optional MongoDB session for transaction support

        Returns:
            GroupProfile or None
        """
        try:
            query_filter = {"group_id": group_id}

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

            # Sort by version descending to get the latest version
            result = await self.model.find_one(
                query_filter, sort=[("version", -1)], session=session
            )

            if result:
                logger.debug(
                    "✅ Successfully retrieved group profile by group ID: %s, version=%s",
                    group_id,
                    result.version,
                )
            else:
                logger.debug("ℹ️  Group profile not found: group_id=%s", group_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve group profile by group ID: %s", e)
            return None

    async def update_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GroupProfile]:
        """
        Update group profile by group ID

        Args:
            group_id: Group ID
            update_data: Update data
            version: Optional version number; if specified, update specific version, otherwise update latest version
            session: Optional MongoDB session for transaction support

        Returns:
            Updated GroupProfile or None
        """
        try:
            # Find the document to update
            if version is not None:
                # Update specific version
                existing_doc = await self.model.find_one(
                    {"group_id": group_id, "version": version}, session=session
                )
            else:
                # Update latest version
                existing_doc = await self.model.find_one(
                    {"group_id": group_id}, sort=[("version", -1)], session=session
                )

            if not existing_doc:
                logger.warning(
                    "⚠️  Group profile not found for update: group_id=%s, version=%s",
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
                "✅ Successfully updated group profile by group ID: group_id=%s, version=%s",
                group_id,
                existing_doc.version,
            )

            return existing_doc
        except Exception as e:
            logger.error("❌ Failed to update group profile by group ID: %s", e)
            return None

    async def delete_by_group_id(
        self,
        group_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Delete group profile by group ID

        Args:
            group_id: Group ID
            version: Optional version number; if specified, delete only that version, otherwise delete all versions
            session: Optional MongoDB session for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            query_filter = {"group_id": group_id}
            if version is not None:
                query_filter["version"] = version

            if version is not None:
                # Delete specific version
                result = await self.model.find_one(query_filter, session=session)
                if not result:
                    logger.warning(
                        "⚠️  Group profile not found for deletion: group_id=%s, version=%s",
                        group_id,
                        version,
                    )
                    return False

                await result.delete(session=session)
                logger.debug(
                    "✅ Successfully deleted group profile by group ID and version: group_id=%s, version=%s",
                    group_id,
                    version,
                )

                # After deletion, ensure latest version flag is correct
                await self.ensure_latest(group_id, session)
                return True
            else:
                # Delete all versions
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ Successfully deleted all group profiles by group ID: group_id=%s, deleted %d records",
                        group_id,
                        deleted_count,
                    )
                else:
                    logger.warning(
                        "⚠️  No group profile found for deletion: group_id=%s", group_id
                    )

                return success
        except Exception as e:
            logger.error("❌ Failed to delete group profile by group ID: %s", e)
            return False

    async def upsert_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        timestamp: Optional[int] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[GroupProfile]:
        """
        Update or insert group profile by group ID

        If update_data contains a version field:
        - If that version exists, update it
        - If that version does not exist, create a new version (version must be provided)
        If update_data does not contain a version field:
        - Get the latest version and update it; if it doesn't exist, raise an error (version must be provided when creating)

        Args:
            group_id: Group ID
            update_data: Data to update (must contain version field when creating a new version)
            timestamp: Timestamp, required when creating a new record
            session: Optional MongoDB session for transaction support

        Returns:
            Updated or created group profile record
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # If version is specified, find that specific version
                existing_doc = await self.model.find_one(
                    {"group_id": group_id, "version": version}, session=session
                )
            else:
                # If version is not specified, find the latest version
                existing_doc = await self.model.find_one(
                    {"group_id": group_id}, sort=[("version", -1)], session=session
                )

            if existing_doc:
                # Update existing record
                for key, value in update_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ Successfully updated existing group profile: group_id=%s, version=%s",
                    group_id,
                    existing_doc.version,
                )

                # If version was updated, ensure latest flag is correct
                if version is not None:
                    await self.ensure_latest(group_id, session)

                return existing_doc
            else:
                # When creating a new record, version must be provided
                if version is None:
                    logger.error(
                        "❌ Version field must be provided when creating a new group profile: group_id=%s",
                        group_id,
                    )
                    raise ValueError(
                        f"Version field must be provided when creating a new group profile: group_id={group_id}"
                    )

                # Create new record, timestamp is required
                if timestamp is None:
                    from time import time

                    timestamp = int(time() * 1000)  # Millisecond timestamp

                new_doc = GroupProfile(
                    group_id=group_id, timestamp=timestamp, **update_data
                )
                await new_doc.create(session=session)
                logger.info(
                    "✅ Successfully created new group profile: group_id=%s, version=%s",
                    group_id,
                    new_doc.version,
                )

                # After creation, ensure latest version flag is correct
                await self.ensure_latest(group_id, session)

                return new_doc
        except ValueError:
            # Re-raise ValueError, do not catch it in Exception
            raise
        except Exception as e:
            logger.error("❌ Failed to update or create group profile: %s", e)
            return None

    # ==================== Query Methods ====================

    async def find_by_group_ids(
        self,
        group_ids: List[str],
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[GroupProfile]:
        """
        Batch retrieve group profiles by list of group IDs

        Args:
            group_ids: List of group IDs
            only_latest: Whether to get only the latest version, default is True. Use is_latest field for filtering in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of GroupProfile
        """
        try:
            if not group_ids:
                return []

            query_filter = {"group_id": {"$in": group_ids}}

            # For batch queries, use is_latest field to filter latest versions
            if only_latest:
                query_filter["is_latest"] = True

            query = self.model.find(query_filter, session=session)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully retrieved group profiles by group ID list: %d group IDs, only_latest=%s, found %d records",
                len(group_ids),
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve group profiles by group ID list: %s", e)
            return []


__all__ = ["GroupProfileRawRepository"]
