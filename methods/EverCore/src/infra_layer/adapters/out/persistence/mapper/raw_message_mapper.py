# -*- coding: utf-8 -*-
"""
RawMessage -> RawData Converter

Handles conversion from RawMessage to RawData.
"""

from typing import Optional, List

from core.observation.logger import get_logger
from common_utils.datetime_utils import from_iso_format
from api_specs.dtos import RawData
from api_specs.request_converter import build_raw_data_from_message
from infra_layer.adapters.out.persistence.document.request.raw_message import RawMessage

logger = get_logger(__name__)


class RawMessageMapper:
    """
    RawMessage -> RawData Converter

    Converts RawMessage document fields into RawData for internal processing.
    """

    @staticmethod
    def to_raw_data(log: RawMessage) -> Optional[RawData]:
        """
        Convert RawMessage to RawData

        Builds RawData from the document's individual fields.
        Uses the stored content_items list directly.

        Args:
            log: RawMessage object

        Returns:
            RawData object or None (if log is None)
        """
        if log is None:
            return None

        # Handle timestamp
        timestamp = None
        if log.timestamp:
            try:
                if isinstance(log.timestamp, str):
                    timestamp = from_iso_format(log.timestamp)
                else:
                    timestamp = log.timestamp
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse timestamp: %s, error: %s", log.timestamp, e
                )
                timestamp = None

        message_id = log.message_id or str(log.id)
        content_items = log.content_items or []

        return build_raw_data_from_message(
            message_id=message_id,
            sender_id=log.sender_id or "",
            content_items=content_items,
            timestamp=timestamp,
            sender_name=log.sender_name,
            role=log.role,
            tool_calls=log.tool_calls,
            tool_call_id=log.tool_call_id,
        )

    @staticmethod
    def to_raw_data_list(logs: List[RawMessage]) -> List[RawData]:
        """
        Batch convert a list of RawMessage objects to a list of RawData objects

        Args:
            logs: List of RawMessage objects

        Returns:
            List of RawData objects (skip records that fail conversion)
        """
        raw_data_list: List[RawData] = []

        for log in logs:
            try:
                raw_data = RawMessageMapper.to_raw_data(log)
                if raw_data:
                    raw_data_list.append(raw_data)
            except (ValueError, TypeError) as e:
                logger.error(
                    "Failed to convert RawMessage to RawData: log_id=%s, error=%s",
                    log.id,
                    e,
                )
                continue

        return raw_data_list
