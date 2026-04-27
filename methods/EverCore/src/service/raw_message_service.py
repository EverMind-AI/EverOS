# -*- coding: utf-8 -*-
"""
Raw Message Service

Directly extract data from MemorizeRequest and save to RawMessage,
replacing the original event listener approach to make timing more controllable.
"""

from typing import List, Optional

from common_utils.datetime_utils import to_iso_format
from core.di import service
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.context.context import get_current_app_info
from core.oxm.constants import MAGIC_ALL
from api_specs.dtos import MemorizeRequest, RawData, RawMessageDTO
from infra_layer.adapters.out.persistence.document.request.raw_message import RawMessage
from infra_layer.adapters.out.persistence.repository.raw_message_repository import (
    RawMessageRepository,
)

logger = get_logger(__name__)


@service("raw_message_service")
class RawMessageService:
    """
    Raw Message Service

    Extract each message from new_raw_data_list in MemorizeRequest and save to RawMessage.
    Return the list of saved message_ids for use in subsequent processes.
    """

    def __init__(self):
        self._repository: Optional[RawMessageRepository] = None

    def _get_repository(self) -> RawMessageRepository:
        """Get Repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(RawMessageRepository)
        return self._repository

    async def save_raw_messages(
        self,
        request: MemorizeRequest,
        version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        method: Optional[str] = None,
        url: Optional[str] = None,
    ) -> List[str]:
        """
        Extract each message from MemorizeRequest and save as individual RawMessage documents.

        Iterates through each RawData in new_raw_data_list, extracts core fields and saves.
        Saved records have sync_status=-1 (pending confirmation).

        Args:
            request: MemorizeRequest object
            version: API version (optional)
            endpoint_name: Endpoint name (optional)
            method: HTTP method (optional)
            url: Request URL (optional)

        Returns:
            List[str]: List of saved message_ids
        """
        if not request.new_raw_data_list:
            logger.debug("new_raw_data_list is empty, skipping save")
            return []

        # Get current request context information
        app_info = get_current_app_info()
        request_id = app_info.get("request_id", "unknown")

        saved_message_ids = []
        repo = self._get_repository()

        session_id = request.session_id

        for raw_data in request.new_raw_data_list:
            try:
                message_id = await self._save_single_raw_data(
                    raw_data=raw_data,
                    group_id=request.group_id,
                    request_id=request_id,
                    session_id=session_id,
                    repo=repo,
                    version=version,
                    endpoint_name=endpoint_name,
                    method=method,
                    url=url,
                    event_id=request_id,  # Use request_id as event_id
                )
                if message_id:
                    saved_message_ids.append(message_id)
            except Exception as e:
                logger.error(
                    "Failed to save RawData to RawMessage: data_id=%s, error=%s",
                    raw_data.data_id,
                    e,
                )

        logger.info(
            "Saved %d raw messages: group_id=%s, message_ids=%s",
            len(saved_message_ids),
            request.group_id,
            saved_message_ids,
        )

        return saved_message_ids

    async def _save_single_raw_data(
        self,
        raw_data: RawData,
        group_id: Optional[str],
        request_id: str,
        session_id: Optional[str] = None,
        repo: RawMessageRepository = None,
        version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        method: Optional[str] = None,
        url: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a single RawData to RawMessage

        Delegates to RawMessageRepository.save_from_raw_data for document
        creation and persistence.

        Args:
            raw_data: RawData object
            group_id: Group ID
            request_id: Request ID
            session_id: Session identifier for conversation isolation
            repo: Repository instance
            version: API version
            endpoint_name: Endpoint name
            method: HTTP method
            url: Request URL
            event_id: Event ID

        Returns:
            Optional[str]: Returns message_id if saved successfully, None otherwise
        """
        if not group_id:
            logger.debug("group_id is empty, skipping save")
            return None

        return await repo.save_from_raw_data(
            raw_data_content=raw_data.content or {},
            data_id=raw_data.data_id,
            group_id=group_id,
            request_id=request_id,
            session_id=session_id,
            version=version,
            endpoint_name=endpoint_name,
            method=method,
            url=url,
            event_id=event_id,
        )

    # ==================== Query Methods ====================

    async def check_duplicate_message(
        self, group_id: str, sender_id: str, message_id: str
    ) -> bool:
        """
        Check if a message with the given group_id, sender_id, and message_id already exists

        Used for duplicate detection before processing new memorize requests.
        This helps prevent duplicate message processing when the same message
        is submitted multiple times.

        Args:
            group_id: Conversation group ID
            sender_id: Sender ID
            message_id: Message ID

        Returns:
            bool: True if the message already exists, False otherwise
        """
        repo = self._get_repository()
        try:
            existing = await repo.find_one_by_group_sender_message(
                group_id=group_id, sender_id=sender_id, message_id=message_id
            )
            if existing:
                logger.info(
                    "Duplicate message detected: group_id=%s, sender_id=%s, message_id=%s",
                    group_id,
                    sender_id,
                    message_id,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to check duplicate message: group_id=%s, sender_id=%s, message_id=%s, error=%s",
                group_id,
                sender_id,
                message_id,
                e,
            )
            # In case of error, return False to allow the request to proceed
            # This is a fail-open approach to avoid blocking legitimate requests
            return False

    async def get_pending_raw_messages(
        self,
        sender_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        sync_status_list: Optional[List[int]] = None,
        limit: int = 1000,
        skip: int = 0,
        ascending: bool = True,
    ) -> List[RawMessage]:
        """
        Get pending (unconsumed) raw messages

        Query raw messages that have not been consumed yet (sync_status=-1 or 0).
        Supports flexible filtering with MAGIC_ALL logic:
        - MAGIC_ALL ("__all__"): Don't filter by this field
        - None or "": Filter for null/empty values
        - Other values: Exact match

        Args:
            sender_id: Sender ID filter
                - MAGIC_ALL: Don't filter by sender_id (default)
                - None or "": Filter for null/empty values
                - Other values: Exact match
            group_ids: List of Group IDs to filter (None to skip filtering, searches all groups)
            sync_status_list: List of sync_status values to filter by
                - Default: [-1, 0] (pending and accumulating, i.e., unconsumed)
                - [-1]: Just log records
                - [0]: In window accumulation
                - [1]: Already fully used
            limit: Maximum number of records to return (default 100)
            skip: Number of records to skip (default 0)
            ascending: If True (default), sort by created_at ascending (oldest first);
                       if False, sort descending (newest first)

        Returns:
            List[RawMessage]: List of pending raw messages
        """
        # Default to unconsumed statuses
        if sync_status_list is None:
            sync_status_list = [-1, 0]

        repo = self._get_repository()

        try:
            results = await repo.find_pending_by_filters(
                sender_id=sender_id,
                group_ids=group_ids,
                sync_status_list=sync_status_list,
                limit=limit,
                skip=skip,
                ascending=ascending,
            )

            logger.debug(
                "Retrieved pending raw messages: sender_id=%s, group_ids=%s, "
                "sync_status_list=%s, count=%d",
                sender_id,
                group_ids,
                sync_status_list,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to get pending raw messages: sender_id=%s, group_ids=%s, error=%s",
                sender_id,
                group_ids,
                e,
            )
            return []

    async def get_pending_messages(
        self,
        sender_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[RawMessageDTO]:
        """
        Get pending (unconsumed) messages as list of RawMessageDTO objects.

        This is a convenience method that wraps get_pending_raw_messages
        and converts the results to RawMessageDTO dataclass instances.

        Args:
            sender_id: Sender ID filter (MAGIC_ALL to skip filtering)
            group_ids: List of Group IDs to filter (None to skip filtering, searches all groups)
            limit: Maximum number of records to return (default 1000)

        Returns:
            List[RawMessageDTO]: List of pending messages
        """
        logs = await self.get_pending_raw_messages(
            sender_id=sender_id, group_ids=group_ids, limit=limit
        )

        # Convert to list of RawMessageDTO
        result = []
        for log in logs:
            pending_msg = RawMessageDTO(
                id=str(log.id),
                request_id=log.request_id,
                message_id=log.message_id,
                group_id=log.group_id,
                session_id=log.session_id,
                sender_id=log.sender_id,
                sender_name=log.sender_name,
                content_items=log.content_items,
                timestamp=log.timestamp,
                created_at=to_iso_format(log.created_at),
                updated_at=to_iso_format(log.updated_at),
            )
            result.append(pending_msg)

        logger.debug(
            "Converted %d pending raw messages to RawMessageDTO: sender_id=%s, group_ids=%s",
            len(result),
            sender_id,
            group_ids,
        )
        return result
