# -*- coding: utf-8 -*-
"""
RawMessage Repository

Raw message data access layer, providing CRUD operations for raw message records.
Used as a replacement for the conversation_data functionality.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.request.raw_message import RawMessage

logger = get_logger(__name__)


@repository("raw_message_repository", primary=True)
class RawMessageRepository(BaseRepository[RawMessage]):
    """
    Raw Message Repository

    Provides CRUD operations and query functionality for raw message records.
    Can be used as an alternative implementation for conversation_data.
    """

    def __init__(self):
        super().__init__(RawMessage)

    # ==================== Save Methods ====================

    async def save(
        self, raw_message: RawMessage, session: Optional[AsyncClientSession] = None
    ) -> Optional[RawMessage]:
        """
        Save raw message

        Args:
            raw_message: RawMessage object
            session: Optional MongoDB session

        Returns:
            Saved RawMessage or None
        """
        try:
            await raw_message.insert(session=session)
            logger.debug(
                "Raw message saved successfully: id=%s, group_id=%s, request_id=%s",
                raw_message.id,
                raw_message.group_id,
                raw_message.request_id,
            )
            return raw_message
        except Exception as e:
            logger.error("Failed to save raw message: %s", e)
            return None

    async def save_from_raw_data(
        self,
        raw_data_content: Dict[str, Any],
        data_id: Optional[str],
        group_id: str,
        request_id: str,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        method: Optional[str] = None,
        url: Optional[str] = None,
        event_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[str]:
        """
        Parse raw data fields, create a RawMessage document, and save it.

        Extracts core message fields (sender, content, timestamps, etc.) from the
        raw data content dict, constructs a RawMessage, and persists it.

        Args:
            raw_data_content: The content dict from RawData (raw_data.content)
            data_id: Message ID (raw_data.message_id)
            group_id: Conversation group ID
            request_id: Request ID
            session_id: Session identifier for conversation isolation
            version: API version
            endpoint_name: Endpoint name
            method: HTTP method
            url: Request URL
            event_id: Event ID
            session: Optional MongoDB session

        Returns:
            Optional[str]: Returns message_id if saved successfully, None otherwise
        """
        content_dict = raw_data_content or {}
        message_id = data_id

        # Extract core message fields
        sender_id = content_dict.get("sender_id", "")
        sender_name = content_dict.get("sender_name") or sender_id
        # Store content_items list directly (e.g. [{type: "text", content: "..."}])
        raw_content = content_dict.get("content")
        content_items = raw_content if isinstance(raw_content, list) else None

        role = content_dict.get("role")
        tool_calls = content_dict.get("tool_calls")
        tool_call_id = content_dict.get("tool_call_id")
        timestamp = self._parse_create_time(
            content_dict.get("timestamp") or content_dict.get("created_at")
        )

        # Create RawMessage document
        raw_message = RawMessage(
            group_id=group_id,
            request_id=request_id,
            session_id=session_id,
            message_id=message_id,
            timestamp=timestamp,
            sender_id=sender_id,
            sender_name=sender_name,
            role=role,
            content_items=content_items,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            version=version,
            endpoint_name=endpoint_name,
            method=method,
            url=url,
            event_id=event_id,
        )

        await self.save(raw_message, session=session)

        logger.debug(
            "Saved raw message from raw data: group_id=%s, message_id=%s",
            group_id,
            message_id,
        )

        return message_id

    @staticmethod
    def _parse_create_time(create_time: Any) -> Optional[str]:
        """Parse creation time and return ISO format string with timezone"""
        if create_time is None:
            return None
        try:
            from common_utils.datetime_utils import to_iso_format

            return to_iso_format(create_time)
        except Exception:
            if isinstance(create_time, str):
                return create_time
            return None

    # ==================== Query Methods ====================

    async def get_by_request_id(
        self, request_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[RawMessage]:
        """
        Get raw message by request ID

        Args:
            request_id: Request ID
            session: Optional MongoDB session

        Returns:
            RawMessage or None
        """
        try:
            result = await RawMessage.find_one(
                {"request_id": request_id}, session=session
            )
            return result
        except Exception as e:
            logger.error("Failed to get raw message by request ID: %s", e)
            return None

    async def find_one_by_group_sender_message(
        self,
        group_id: str,
        sender_id: str,
        message_id: str,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[RawMessage]:
        """
        Find a single raw message by group_id, sender_id, and message_id

        Used for duplicate detection before saving new raw messages.
        Uses composite index (group_id, sender_id, message_id) for efficient lookup.

        Args:
            group_id: Conversation group ID
            sender_id: Sender ID
            message_id: Message ID
            session: Optional MongoDB session

        Returns:
            RawMessage if found, None otherwise
        """
        try:
            result = await RawMessage.find_one(
                {
                    "group_id": group_id,
                    "sender_id": sender_id,
                    "message_id": message_id,
                },
                session=session,
            )
            if result:
                logger.debug(
                    "Found existing raw message: group_id=%s, sender_id=%s, message_id=%s",
                    group_id,
                    sender_id,
                    message_id,
                )
            return result
        except Exception as e:
            logger.error(
                "Failed to find raw message by group_id/sender_id/message_id: "
                "group_id=%s, sender_id=%s, message_id=%s, error=%s",
                group_id,
                sender_id,
                message_id,
                e,
            )
            return None

    async def find_by_group_id(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        sync_status: Optional[int] = 0,
        session: Optional[AsyncClientSession] = None,
    ) -> List[RawMessage]:
        """
        Query raw messages by group_id

        Args:
            group_id: Conversation group ID
            session_id: Session identifier (optional, filters by session when provided)
            start_time: Start time
            end_time: End time
            limit: Maximum number of records to return
            sync_status: Sync status filter (default 0=in window accumulation, None=no filter)
            session: Optional MongoDB session

        Returns:
            List of RawMessage
        """
        try:
            query = {"group_id": group_id}

            if session_id is not None:
                query["session_id"] = session_id

            # Filter by status
            if sync_status is not None:
                query["sync_status"] = sync_status

            if start_time:
                query["created_at"] = {"$gte": start_time}
            if end_time:
                if "created_at" in query:
                    query["created_at"]["$lte"] = end_time
                else:
                    query["created_at"] = {"$lte": end_time}

            results = (
                await RawMessage.find(query, session=session)
                .sort([("created_at", 1)])
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "Query raw messages by group_id: group_id=%s, sync_status=%s, count=%d",
                group_id,
                sync_status,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("Failed to query raw messages by group_id: %s", e)
            return []

    async def find_by_group_id_with_statuses(
        self,
        group_id: str,
        sync_status_list: List[int],
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        ascending: bool = True,
        exclude_message_ids: Optional[List[str]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[RawMessage]:
        """
        Query raw messages by group_id with multiple sync_status values

        Args:
            group_id: Conversation group ID
            sync_status_list: List of sync_status values to filter by
            session_id: Session identifier (optional)
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of records to return
            ascending: Sort ascending by created_at (default True)
            exclude_message_ids: Message IDs to exclude from results
            session: Optional MongoDB session

        Returns:
            List of RawMessage
        """
        try:
            query = {"group_id": group_id}

            if session_id is not None:
                query["session_id"] = session_id

            if sync_status_list:
                if len(sync_status_list) == 1:
                    query["sync_status"] = sync_status_list[0]
                else:
                    query["sync_status"] = {"$in": sync_status_list}

            if start_time:
                query["created_at"] = {"$gte": start_time}
            if end_time:
                if "created_at" in query:
                    query["created_at"]["$lte"] = end_time
                else:
                    query["created_at"] = {"$lte": end_time}

            if exclude_message_ids:
                query["message_id"] = {"$nin": exclude_message_ids}

            sort_order = 1 if ascending else -1

            results = (
                await RawMessage.find(query, session=session)
                .sort([("created_at", sort_order)])
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "Query raw messages by group_id with statuses: group_id=%s, sync_status_list=%s, count=%d",
                group_id,
                sync_status_list,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to query raw messages by group_id with statuses: %s", e
            )
            return []

    async def find_by_sender_id(
        self,
        sender_id: str,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[RawMessage]:
        """
        Query raw messages by sender ID

        Args:
            sender_id: Sender ID
            limit: Maximum number of records to return
            session: Optional MongoDB session

        Returns:
            List of RawMessage
        """
        try:
            results = (
                await RawMessage.find({"sender_id": sender_id}, session=session)
                .sort([("created_at", -1)])
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "Query raw messages by sender_id: sender_id=%s, count=%d",
                sender_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("Failed to query raw messages by sender_id: %s", e)
            return []

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Delete raw messages by group_id

        Args:
            group_id: Conversation group ID
            session: Optional MongoDB session

        Returns:
            Number of deleted records
        """
        try:
            result = await RawMessage.find(
                {"group_id": group_id}, session=session
            ).delete()
            deleted_count = result.deleted_count if result else 0
            logger.info(
                "Deleted raw messages: group_id=%s, deleted=%d", group_id, deleted_count
            )
            return deleted_count
        except Exception as e:
            logger.error(
                "Failed to delete raw messages: group_id=%s, error=%s", group_id, e
            )
            return 0

    async def delete_by_filters(
        self,
        sender_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete raw messages by filter conditions

        Args:
            sender_id: Sender ID filter
                - MAGIC_ALL ("__all__"): Don't filter by sender_id
                - Other values: Exact match
            group_id: Group ID filter
            session: Optional MongoDB session

        Returns:
            Number of soft-deleted records
        """
        filter_dict = {}

        if sender_id != MAGIC_ALL:
            if sender_id == "" or sender_id is None:
                filter_dict["sender_id"] = {"$in": [None, ""]}
            else:
                filter_dict["sender_id"] = sender_id

        if group_id != MAGIC_ALL:
            if group_id is None or group_id == "":
                filter_dict["group_id"] = {"$in": [None, ""]}
            else:
                filter_dict["group_id"] = group_id

        if not filter_dict:
            logger.warning("No filter conditions provided for delete_by_filters")
            return 0

        try:
            result = await self.model.delete_many(filter_dict, session=session)
            count = result.modified_count if result else 0
            logger.info(
                "Soft deleted raw messages: filter=%s, deleted=%d", filter_dict, count
            )
            return count
        except Exception as e:
            logger.error(
                "Failed to soft delete raw messages: filter=%s, error=%s",
                {"sender_id": sender_id, "group_id": group_id},
                e,
            )
            return 0

    # ==================== Sync Status Management ====================
    # sync_status state transitions:
    # -1 (log record) -> 0 (window accumulation) -> 1 (used)

    async def confirm_accumulation_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Confirm records for the specified group_id as window accumulation state

        Batch update sync_status: -1 -> 0.
        Uses (group_id, sync_status) composite index for efficient querying.

        Args:
            group_id: Conversation group ID
            session: Optional MongoDB session

        Returns:
            Number of updated records
        """
        try:
            collection = RawMessage.get_pymongo_collection()
            result = await collection.update_many(
                {"group_id": group_id, "sync_status": -1},
                {"$set": {"sync_status": 0}},
                session=session,
            )
            modified_count = result.modified_count if result else 0
            logger.info(
                "Confirmed window accumulation: group_id=%s, modified=%d",
                group_id,
                modified_count,
            )
            return modified_count
        except Exception as e:
            logger.error(
                "Failed to confirm window accumulation: group_id=%s, error=%s",
                group_id,
                e,
            )
            return 0

    async def confirm_accumulation_by_message_ids(
        self,
        group_id: str,
        message_ids: List[str],
        session_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Confirm records for the specified message_id list as window accumulation state

        Precise update: only update records with specified message_id.
        sync_status: -1 -> 0

        Args:
            group_id: Conversation group ID
            message_ids: List of message_ids to update
            session_id: Session identifier (optional)
            session: Optional MongoDB session

        Returns:
            Number of updated records
        """
        if not message_ids:
            logger.debug("message_ids is empty, skipping update")
            return 0

        try:
            collection = RawMessage.get_pymongo_collection()
            query = {
                "group_id": group_id,
                "message_id": {"$in": message_ids},
                "sync_status": -1,
            }
            if session_id is not None:
                query["session_id"] = session_id
            result = await collection.update_many(
                query, {"$set": {"sync_status": 0}}, session=session
            )
            modified_count = result.modified_count if result else 0
            logger.info(
                "Confirmed window accumulation (precise): group_id=%s, message_ids=%d, modified=%d",
                group_id,
                len(message_ids),
                modified_count,
            )
            return modified_count
        except Exception as e:
            logger.error(
                "Failed to confirm window accumulation (precise): group_id=%s, error=%s",
                group_id,
                e,
            )
            return 0

    async def mark_as_used_by_group_id(
        self,
        group_id: str,
        session_id: Optional[str] = None,
        exclude_message_ids: Optional[List[str]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Mark all pending and accumulating data for the specified group_id as used

        Batch update sync_status: -1 or 0 -> 1 (after boundary detection).

        Args:
            group_id: Conversation group ID
            session_id: Session identifier (optional)
            exclude_message_ids: Message IDs to exclude from update
            session: Optional MongoDB session

        Returns:
            Number of updated records
        """
        try:
            collection = RawMessage.get_pymongo_collection()
            query = {"group_id": group_id, "sync_status": {"$in": [-1, 0]}}

            if session_id is not None:
                query["session_id"] = session_id

            if exclude_message_ids:
                query["message_id"] = {"$nin": exclude_message_ids}

            result = await collection.update_many(
                query, {"$set": {"sync_status": 1}}, session=session
            )
            modified_count = result.modified_count if result else 0
            logger.info(
                "Marked as used: group_id=%s, exclude=%d, modified=%d",
                group_id,
                len(exclude_message_ids) if exclude_message_ids else 0,
                modified_count,
            )
            return modified_count
        except Exception as e:
            logger.error("Failed to mark as used: group_id=%s, error=%s", group_id, e)
            return 0

    # ==================== Flexible Query Methods ====================

    async def find_pending_by_filters(
        self,
        sender_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        sync_status_list: Optional[List[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        skip: int = 0,
        ascending: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[RawMessage]:
        """
        Query pending raw messages by flexible filters

        Supports MAGIC_ALL logic:
        - MAGIC_ALL ("__all__"): Don't filter by this field
        - None or "": Filter for null/empty values
        - Other values: Exact match

        Args:
            sender_id: Sender ID filter
                - MAGIC_ALL: Don't filter by sender_id
                - None or "": Filter for null/empty values
                - Other values: Exact match
            group_ids: List of Group IDs to filter (None = search all groups)
            sync_status_list: List of sync_status values to filter by
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of records to return
            skip: Number of records to skip
            ascending: Sort ascending by created_at (default True)
            session: Optional MongoDB session

        Returns:
            List of RawMessage
        """
        if sync_status_list is None:
            sync_status_list = [-1, 0]

        try:
            query = {}

            if sender_id != MAGIC_ALL:
                if sender_id == "" or sender_id is None:
                    query["sender_id"] = {"$in": [None, ""]}
                else:
                    query["sender_id"] = sender_id

            if group_ids is not None and len(group_ids) > 0:
                query["group_id"] = {"$in": group_ids}

            if sync_status_list:
                if len(sync_status_list) == 1:
                    query["sync_status"] = sync_status_list[0]
                else:
                    query["sync_status"] = {"$in": sync_status_list}

            if start_time is not None or end_time is not None:
                time_filter = {}
                if start_time is not None:
                    time_filter["$gte"] = start_time
                if end_time is not None:
                    time_filter["$lte"] = end_time
                query["created_at"] = time_filter

            sort_order = 1 if ascending else -1

            results = (
                await RawMessage.find(query, session=session)
                .sort([("created_at", sort_order)])
                .skip(skip)
                .limit(limit)
                .to_list()
            )

            logger.debug(
                "Query pending raw messages: sender_id=%s, group_ids=%s, "
                "sync_status_list=%s, skip=%d, limit=%d, count=%d",
                sender_id,
                group_ids,
                sync_status_list,
                skip,
                limit,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to query pending raw messages: sender_id=%s, group_ids=%s, error=%s",
                sender_id,
                group_ids,
                e,
            )
            return []
