from typing import Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
    ConversationStatus,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("conversation_status_raw_repository", primary=True)
class ConversationStatusRawRepository(BaseRepository[ConversationStatus]):
    """
    Conversation status raw data repository

    Provides CRUD operations and query capabilities for conversation status data.
    """

    def __init__(self):
        super().__init__(ConversationStatus)

    # ==================== Basic CRUD Operations ====================

    async def get_by_group_id(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[ConversationStatus]:
        """Get conversation status by group ID and session ID"""
        try:
            query = {"group_id": group_id}
            if session_id is not None:
                query["session_id"] = session_id
            result = await self.model.find_one(query, session=session)
            if result:
                logger.debug(
                    "✅ Successfully retrieved conversation status by group ID: %s",
                    group_id,
                )
            else:
                logger.debug("⚠️  Conversation status not found: group_id=%s", group_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve conversation status by group ID: %s", e)
            return None

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """Delete conversation status by group ID"""
        try:
            result = await self.model.find_one({"group_id": group_id}, session=session)
            if not result:
                logger.warning(
                    "⚠️  Conversation status to delete not found: group_id=%s", group_id
                )
                return False

            await result.delete(session=session)
            logger.info(
                "✅ Successfully deleted conversation status by group ID: %s", group_id
            )
            return True
        except Exception as e:
            logger.error("❌ Failed to delete conversation status by group ID: %s", e)
            return False

    async def upsert_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        session_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[ConversationStatus]:
        """Update or insert conversation status by group ID and session ID

        Uses MongoDB atomic upsert operation to avoid concurrency race conditions.
        If a matching record is found, it updates it; otherwise, it creates a new record.
        Unique key is (group_id, session_id).

        Args:
            group_id: Group ID
            update_data: Data to update
            session_id: Session identifier for conversation isolation
            session: MongoDB session

        Returns:
            The updated or created conversation status record
        """
        try:
            # Build query with session_id
            query = {"group_id": group_id}
            if session_id is not None:
                query["session_id"] = session_id

            # 1. First try to find an existing record
            existing_doc = await self.model.find_one(query, session=session)

            if existing_doc:
                # Record found, update directly
                for key, value in update_data.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "Successfully updated existing conversation status: group_id=%s, session_id=%s",
                    group_id,
                    session_id,
                )
                return existing_doc

            # 2. Record not found, try to create a new one
            try:
                create_data = {**update_data}
                if session_id is not None:
                    create_data["session_id"] = session_id
                new_doc = ConversationStatus(group_id=group_id, **create_data)
                await new_doc.create(session=session)
                logger.info(
                    "Successfully created new conversation status: group_id=%s, session_id=%s",
                    group_id,
                    session_id,
                )
                return new_doc

            except Exception as create_error:
                # 3. Creation failed, check if it's a duplicate key error (concurrent case)
                error_str = str(create_error)
                if "E11000" in error_str and "duplicate key" in error_str:
                    logger.warning(
                        "Concurrent creation conflict, re-lookup and update: group_id=%s, session_id=%s",
                        group_id,
                        session_id,
                    )

                    # Duplicate key error means another thread has already created the record, re-lookup and update
                    retry_doc = await self.model.find_one(query, session=session)

                    if retry_doc:
                        # Found the record created by another thread, update it
                        for key, value in update_data.items():
                            setattr(retry_doc, key, value)
                        await retry_doc.save(session=session)
                        logger.debug(
                            "Successfully updated after concurrency conflict: group_id=%s, session_id=%s",
                            group_id,
                            session_id,
                        )
                        return retry_doc
                    else:
                        logger.error(
                            "Still unable to find record after concurrency conflict: group_id=%s, session_id=%s",
                            group_id,
                            session_id,
                        )
                        return None
                else:
                    # Other types of creation errors, re-raise
                    raise create_error

        except Exception as e:
            logger.error("Failed to update or create conversation status: %s", e)
            return None

    # ==================== Statistics Methods ====================

    async def count_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """Count the number of conversation statuses for a specified group"""
        try:
            count = await self.model.find(
                {"group_id": group_id}, session=session
            ).count()
            logger.debug(
                "✅ Successfully counted conversation statuses: group_id=%s, count=%d",
                group_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count conversation statuses: %s", e)
            return 0
