# -*- coding: utf-8 -*-
"""
ConversationDataRepository interface and implementation

Conversation data storage based on RawMessage, replacing the original Redis implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.observation.logger import get_logger
from core.di.decorators import repository
from core.di import get_bean
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from common_utils.datetime_utils import from_iso_format
from infra_layer.adapters.out.persistence.repository.raw_message_repository import (
    RawMessageRepository,
)
from infra_layer.adapters.out.persistence.mapper.raw_message_mapper import (
    RawMessageMapper,
)

logger = get_logger(__name__)


# ==================== Interface Definition ====================


class ConversationDataRepository(ABC):
    """Conversation data access interface"""

    @abstractmethod
    async def save_conversation_data(
        self,
        raw_data_list: List[RawData],
        group_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Confirm conversation data enters window accumulation

        Updates sync_status=-1 to sync_status=0 for records matching message_ids in raw_data_list.

        Args:
            raw_data_list: List of RawData, data_id is used to identify which records to update
            group_id: Group ID
            session_id: Session identifier for conversation isolation

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_conversation_data(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[RawData]:
        """
        Get conversation data (sync_status=-1 or 0)

        Returns both pending (-1) and accumulating (0) records.

        Args:
            group_id: Group ID
            session_id: Session identifier for conversation isolation
            start_time: Start time (ISO format string)
            end_time: End time (ISO format string)
            limit: Maximum number of records to return
            exclude_message_ids: Message IDs to exclude from results

        Returns:
            List[RawData]: List of conversation data
        """
        pass

    @abstractmethod
    async def delete_conversation_data(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Mark all pending and accumulating data as used

        Updates sync_status=-1 and 0 to sync_status=1.

        Args:
            group_id: Group ID
            session_id: Session identifier for conversation isolation
            exclude_message_ids: Message IDs to exclude from update

        Returns:
            bool: Return True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def fetch_unprocessed_conversation_data(
        self, group_id: str, session_id: Optional[str] = None, limit: int = 100
    ) -> List[RawData]:
        """
        Fetch unprocessed conversation data (sync_status=-1 or 0)

        Unlike get_conversation_data, this method does not have time range filters
        and returns results in ascending order (oldest first).

        Args:
            group_id: Group ID
            session_id: Session identifier for conversation isolation
            limit: Maximum number of records to return (in ascending order)

        Returns:
            List[RawData]: List of unprocessed conversation data
        """
        pass


# ==================== Implementation ====================


@repository("conversation_data_raw_repository", primary=True)
class ConversationDataRepositoryImpl(ConversationDataRepository):
    """
    ConversationDataRepository implementation based on RawMessage

    Reuses RawMessage storage for conversation data, converting between RawData
    and RawMessage. Data is automatically saved to RawMessage through
    the RequestHistoryEvent listener.
    """

    def __init__(self):
        self._repo: Optional[RawMessageRepository] = None

    def _get_repo(self) -> RawMessageRepository:
        """Lazy load RawMessageRepository"""
        if self._repo is None:
            self._repo = get_bean("raw_message_repository")
        return self._repo

    # ==================== ConversationDataRepository Interface Implementation ====================

    async def save_conversation_data(
        self,
        raw_data_list: List[RawData],
        group_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Confirm conversation data enters window accumulation

        Updates sync_status=-1 to sync_status=0 for records matching the message_ids
        in raw_data_list. Only confirms the specific messages provided.

        sync_status state transitions:
        - -1: Just a log record (raw request just saved via listener)
        -  0: In window accumulation (confirmed via this method)
        -  1: Already fully used (marked via delete_conversation_data)

        Args:
            raw_data_list: RawData list, data_id is used to identify which records to update
            group_id: Conversation group ID
            session_id: Session identifier for conversation isolation

        Returns:
            bool: True if operation succeeds, False otherwise
        """
        logger.info(
            "Confirming conversation data enters window accumulation: group_id=%s, session_id=%s, data_count=%d",
            group_id,
            session_id,
            len(raw_data_list) if raw_data_list else 0,
        )

        try:
            repo = self._get_repo()

            # Extract message_id list (filter out empty values)
            message_ids = [r.data_id for r in (raw_data_list or []) if r.data_id]

            if not message_ids:
                logger.debug("No message_ids to confirm, skipping update")
                return True

            # Precise update: only update records with specified message_id
            modified_count = await repo.confirm_accumulation_by_message_ids(
                group_id, message_ids, session_id=session_id
            )

            logger.info(
                "Window accumulation confirmation completed: group_id=%s, session_id=%s, message_ids=%d, modified=%d",
                group_id,
                session_id,
                len(message_ids),
                modified_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Window accumulation confirmation failed: group_id=%s, session_id=%s, error=%s",
                group_id,
                session_id,
                e,
            )
            return False

    async def get_conversation_data(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[RawData]:
        """
        Get conversation data (sync_status=-1 or 0)

        Queries RawMessage with sync_status=-1 (pending) or 0 (accumulating)
        and converts to RawData. Returns both pending and accumulating records.

        sync_status state description:
        - -1: Pending (returned)
        -  0: In window accumulation (returned)
        -  1: Already fully used (not returned)

        Args:
            group_id: Conversation group ID
            session_id: Session identifier for conversation isolation
            start_time: Start time (ISO format string)
            end_time: End time (ISO format string)
            limit: Maximum number of records to return
            exclude_message_ids: Message IDs to exclude from results

        Returns:
            List[RawData]: List of conversation data
        """
        logger.info(
            "Fetching conversation data: group_id=%s, session_id=%s, start_time=%s, end_time=%s, limit=%d, exclude=%d",
            group_id,
            session_id,
            start_time,
            end_time,
            limit,
            len(exclude_message_ids) if exclude_message_ids else 0,
        )

        try:
            repo = self._get_repo()

            # Convert time format
            start_dt = from_iso_format(start_time) if start_time else None
            end_dt = from_iso_format(end_time) if end_time else None

            # Query RawMessage with sync_status=-1 or 0
            logs = await repo.find_by_group_id_with_statuses(
                group_id=group_id,
                sync_status_list=[-1, 0],
                session_id=session_id,
                start_time=start_dt,
                end_time=end_dt,
                limit=limit,
                exclude_message_ids=exclude_message_ids,
            )

            # Use mapper to convert to RawData list
            raw_data_list = RawMessageMapper.to_raw_data_list(logs)

            logger.info(
                "Conversation data fetch completed: group_id=%s, session_id=%s, count=%d",
                group_id,
                session_id,
                len(raw_data_list),
            )
            return raw_data_list

        except Exception as e:
            logger.error(
                "Conversation data fetch failed: group_id=%s, session_id=%s, error=%s",
                group_id,
                session_id,
                e,
            )
            return []

    async def delete_conversation_data(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Mark all pending and accumulating data as used

        Updates sync_status=-1 (pending) and 0 (accumulating) to sync_status=1 (used).
        This marks all conversation data for the group as fully processed.

        Args:
            group_id: Conversation group ID
            session_id: Session identifier for conversation isolation
            exclude_message_ids: Message IDs to exclude from update

        Returns:
            bool: True if operation succeeds, False otherwise
        """
        logger.info(
            "Marking conversation data as used: group_id=%s, session_id=%s, exclude=%d",
            group_id,
            session_id,
            len(exclude_message_ids) if exclude_message_ids else 0,
        )

        try:
            repo = self._get_repo()
            # Update sync_status: -1,0 -> 1
            modified_count = await repo.mark_as_used_by_group_id(
                group_id, session_id=session_id, exclude_message_ids=exclude_message_ids
            )

            logger.info(
                "Conversation data marked as used: group_id=%s, session_id=%s, modified=%d",
                group_id,
                session_id,
                modified_count,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to mark conversation data as used: group_id=%s, session_id=%s, error=%s",
                group_id,
                session_id,
                e,
            )
            return False

    async def fetch_unprocessed_conversation_data(
        self, group_id: str, session_id: Optional[str] = None, limit: int = 100
    ) -> List[RawData]:
        """
        Fetch unprocessed conversation data (sync_status=-1 or 0)

        Unlike get_conversation_data, this method:
        - Does not have start_time and end_time filters
        - Returns results in ascending order (oldest first) with limit applied

        This is useful for fetching all pending/accumulating messages that need
        to be processed, without time range restrictions.

        Args:
            group_id: Conversation group ID
            session_id: Session identifier for conversation isolation
            limit: Maximum number of records to return (in ascending order, oldest first)

        Returns:
            List[RawData]: List of unprocessed conversation data
        """
        logger.info(
            "Fetching unprocessed conversation data: group_id=%s, session_id=%s, limit=%d",
            group_id,
            session_id,
            limit,
        )

        try:
            repo = self._get_repo()

            # Query both pending (-1) and accumulating (0) records
            # No time range filter, ascending order (oldest first)
            logs = await repo.find_by_group_id_with_statuses(
                group_id=group_id,
                sync_status_list=[-1, 0],
                session_id=session_id,
                start_time=None,
                end_time=None,
                limit=limit,
                ascending=True,
            )

            # Use mapper to convert to RawData list
            raw_data_list = RawMessageMapper.to_raw_data_list(logs)

            logger.info(
                "Unprocessed conversation data fetch completed: group_id=%s, session_id=%s, count=%d",
                group_id,
                session_id,
                len(raw_data_list),
            )
            return raw_data_list

        except Exception as e:
            logger.error(
                "Unprocessed conversation data fetch failed: group_id=%s, session_id=%s, error=%s",
                group_id,
                session_id,
                e,
            )
            return []
