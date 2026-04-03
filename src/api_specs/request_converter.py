"""
Request converter module

This module contains various functions to convert external request formats to internal Request objects.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from api_specs.dtos import FetchMemRequest, MemorizeRequest, RawData, RetrieveMemRequest
from api_specs.memory_models import MemoryType, RetrieveMethod
from api_specs.memory_types import RawDataType
from common_utils.datetime_utils import from_iso_format
from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL

logger = get_logger(__name__)


def generate_single_user_group_id(sender: str) -> str:
    """
    Generate a group_id for single-user mode based on sender (user_id) hash.

    This function creates a deterministic group_id by hashing the sender
    and appending '_group' suffix. This is used when group_id is not provided,
    representing single-user mode where each user's messages are extracted
    into separate memory spaces.

    Args:
        sender: The sender user ID (equivalent to user_id internally)

    Returns:
        str: Generated group_id in format: {hash(sender)[:16]}_group
    """
    # Use MD5 hash for deterministic and compact result
    hash_value = hashlib.md5(sender.encode('utf-8')).hexdigest()[:16]
    return f"{hash_value}_group"


class DataFields:
    """Data field constants"""

    MESSAGES = "messages"
    RAW_DATA_TYPE = "raw_data_type"
    GROUP_ID = "group_id"


def _strip_if_str(value: Any) -> Any:
    """Normalize string input by trimming leading/trailing whitespace."""
    if isinstance(value, str):
        return value.strip()
    return value


def _parse_memory_type(value: Any) -> MemoryType:
    """Parse input value into MemoryType with string normalization."""
    if isinstance(value, MemoryType):
        return value
    return MemoryType(_strip_if_str(value))


def _parse_retrieve_method(value: Any) -> RetrieveMethod:
    """Parse input value into RetrieveMethod with a descriptive error."""
    if isinstance(value, RetrieveMethod):
        return value
    normalized = _strip_if_str(value)
    try:
        return RetrieveMethod(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid retrieve_method: {normalized}. "
            f"Supported methods: {[m.value for m in RetrieveMethod]}"
        ) from exc


def _parse_int(value: Any, default: int) -> int:
    """Parse integer values from query/body payloads."""
    if value is None:
        return default
    normalized = _strip_if_str(value)
    return int(normalized)


def _parse_float(value: Any) -> Optional[float]:
    """Parse optional float values from query/body payloads."""
    if value is None:
        return None
    normalized = _strip_if_str(value)
    return float(normalized)


def _parse_bool(value: Any, default: bool) -> bool:
    """Parse optional bool values from query/body payloads."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in ("true", "1", "yes")
    return bool(value)


def _parse_memory_types(raw_memory_types: Any) -> List[MemoryType]:
    """Parse memory_types payload into a normalized MemoryType list."""
    if raw_memory_types is None:
        raw_items: List[Any] = []
    elif isinstance(raw_memory_types, str):
        raw_items = [
            mt.strip() for mt in raw_memory_types.split(",") if mt and mt.strip()
        ]
    elif isinstance(raw_memory_types, list):
        raw_items = raw_memory_types
    else:
        raw_items = [raw_memory_types]

    memory_types: List[MemoryType] = []
    for raw_item in raw_items:
        if isinstance(raw_item, MemoryType):
            memory_types.append(raw_item)
            continue
        if not isinstance(raw_item, str):
            continue

        normalized = raw_item.strip()
        if not normalized:
            continue

        try:
            memory_types.append(MemoryType(normalized))
        except ValueError:
            logger.error(f"Invalid memory_type: {raw_item}, skipping")

    if not memory_types:
        return [MemoryType.EPISODIC_MEMORY]
    return memory_types


def convert_dict_to_fetch_mem_request(data: Dict[str, Any]) -> FetchMemRequest:
    """
    Convert dictionary to FetchMemRequest object

    Args:
        data: Dictionary containing FetchMemRequest fields

    Returns:
        FetchMemRequest object

    Raises:
        ValueError: When required fields are missing or have incorrect types
    """
    try:
        memory_type = _parse_memory_type(
            data.get("memory_type", MemoryType.EPISODIC_MEMORY.value)
        )
        logger.debug(f"version_range: {data.get('version_range', None)}")

        limit = _parse_int(data.get("limit"), default=10)
        offset = _parse_int(data.get("offset"), default=0)

        # Build FetchMemRequest object
        return FetchMemRequest(
            user_id=data.get(
                "user_id", MAGIC_ALL
            ),  # User ID, use MAGIC_ALL to skip user filtering
            group_id=data.get(
                "group_id", MAGIC_ALL
            ),  # Group ID, use MAGIC_ALL to skip group filtering
            memory_type=memory_type,
            limit=limit,
            offset=offset,
            version_range=data.get("version_range", None),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
        )
    except Exception as exc:
        raise ValueError(f"FetchMemRequest conversion failed: {exc}") from exc


def convert_dict_to_retrieve_mem_request(
    data: Dict[str, Any], query: Optional[str] = None
) -> RetrieveMemRequest:
    """
    Convert dictionary to RetrieveMemRequest object

    Args:
        data: Dictionary containing RetrieveMemRequest fields
        query: Query text (optional)

    Returns:
        RetrieveMemRequest object

    Raises:
        ValueError: When required fields are missing or have incorrect types
    """
    try:
        retrieve_method = _parse_retrieve_method(
            data.get("retrieve_method", RetrieveMethod.KEYWORD.value)
        )
        logger.debug(f"[DEBUG] converted retrieve_method: {retrieve_method}")

        top_k = _parse_int(data.get("top_k"), default=10)
        include_metadata = _parse_bool(data.get("include_metadata"), default=True)
        radius = _parse_float(data.get("radius"))
        memory_types = _parse_memory_types(data.get("memory_types", []))

        return RetrieveMemRequest(
            retrieve_method=retrieve_method,
            user_id=data.get(
                "user_id", MAGIC_ALL
            ),  # User ID, use MAGIC_ALL to skip user filtering
            group_id=data.get(
                "group_id", MAGIC_ALL
            ),  # Group ID, use MAGIC_ALL to skip group filtering
            query=query or data.get("query", None),
            memory_types=memory_types,
            top_k=top_k,
            include_metadata=include_metadata,
            start_time=data.get("start_time", None),
            end_time=data.get("end_time", None),
            radius=radius,  # COSINE similarity threshold
        )
    except Exception as exc:
        raise ValueError(f"RetrieveMemRequest conversion failed: {exc}") from exc


# =========================================


def normalize_refer_list(refer_list: List[Any]) -> List[str]:
    """
    Normalize refer_list format to a list of message IDs

    Supports two formats:
    1. String list: ["msg_id_1", "msg_id_2"]
    2. MessageReference object list: [{"message_id": "msg_id_1", ...}, ...]

    Args:
        refer_list: Original reference list

    Returns:
        List[str]: Normalized list of message IDs
    """
    if not refer_list:
        return []

    normalized: List[str] = []
    for refer in refer_list:
        if isinstance(refer, str):
            normalized.append(refer)
        elif isinstance(refer, dict):
            ref_msg_id = refer.get("message_id")
            if ref_msg_id:
                normalized.append(str(ref_msg_id))
    return normalized


def build_raw_data_from_simple_message(
    message_id: str,
    sender: str,
    content: str,
    timestamp: datetime,
    sender_name: Optional[str] = None,
    role: Optional[str] = None,
    group_id: Optional[str] = None,
    group_name: Optional[str] = None,
    refer_list: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> RawData:
    """
    Build RawData object from simple message fields.

    This is the canonical function for creating RawData from simple message format.
    All code that needs to create RawData from simple messages should use this function
    to ensure consistency.

    Args:
        message_id: Message ID (required)
        sender: Sender user ID (required)
        content: Message content (required)
        timestamp: Message timestamp as datetime object (required)
        sender_name: Sender display name (defaults to sender if not provided)
        role: Message sender role, "user" for human or "assistant" for AI (optional)
        group_id: Group ID (optional)
        group_name: Group name (optional)
        refer_list: Normalized list of referenced message IDs (optional)
        extra_metadata: Additional metadata to merge (optional)

    Returns:
        RawData: Fully constructed RawData object
    """
    # Use sender as sender_name if not provided
    if sender_name is None:
        sender_name = sender

    # Ensure refer_list is a list
    if refer_list is None:
        refer_list = []

    # Build content dictionary with all required fields
    raw_content = {
        "speaker_name": sender_name,
        "role": role,  # Message sender role: "user" or "assistant"
        "receiverId": None,
        "roomId": group_id,
        "groupName": group_name,
        "userIdList": [],
        "referList": refer_list,
        "content": content,
        "timestamp": timestamp,
        "createBy": sender,
        "updateTime": timestamp,
        "orgId": None,
        "speaker_id": sender,
        "msgType": 1,  # TEXT
        "data_id": message_id,
    }

    # Build metadata
    metadata = {
        "original_id": message_id,
        "createTime": timestamp,
        "updateTime": timestamp,
        "createBy": sender,
        "orgId": None,
    }

    # Merge extra metadata if provided
    if extra_metadata:
        metadata.update(extra_metadata)

    return RawData(content=raw_content, data_id=message_id, metadata=metadata)


async def convert_simple_message_to_memorize_request(
    message_data: Dict[str, Any],
) -> MemorizeRequest:
    """
    Convert simple direct single message format directly to MemorizeRequest

    This is a unified conversion function that combines the previous two-step conversion
    (convert_simple_message_to_memorize_input + handle_conversation_format) into one.

    Args:
        message_data: Simple single message data, containing:
            - sender (required): Sender user ID (also used as user_id internally)
            - group_id (optional): Group ID. If not provided, will auto-generate based on
              hash(sender) + '_group' suffix for single-user mode
            - group_name (optional): Group name
            - message_id (required): Message ID
            - create_time (required): Creation time (ISO 8601 format)
            - sender_name (optional): Sender name
            - role (optional): Message sender role ("user" for human, "assistant" for AI)
            - content (required): Message content
            - refer_list (optional): List of referenced message IDs

    Returns:
        MemorizeRequest: Ready-to-use memorize request object

    Raises:
        ValueError: When required fields are missing
    """
    # Extract fields
    group_id = message_data.get("group_id")
    group_name = message_data.get("group_name")
    message_id = message_data.get("message_id")
    create_time_str = message_data.get("create_time")
    sender = message_data.get("sender")
    sender_name = message_data.get("sender_name", sender)
    role = message_data.get("role")  # "user" or "assistant"
    content = message_data.get("content", "")
    refer_list = message_data.get("refer_list", [])

    # Validate required fields
    if not sender:
        raise ValueError("Missing required field: sender")
    if not message_id:
        raise ValueError("Missing required field: message_id")
    if not create_time_str:
        raise ValueError("Missing required field: create_time")
    if not content:
        raise ValueError("Missing required field: content")

    # Auto-generate group_id if not provided (single-user mode)
    if not group_id:
        group_id = generate_single_user_group_id(sender)
        logger.debug(
            f"Auto-generated group_id for single-user mode: {group_id} (sender: {sender})"
        )

    # Normalize refer_list
    normalized_refer_list = normalize_refer_list(refer_list)

    # Parse timestamp
    timestamp = from_iso_format(create_time_str, ZoneInfo("UTC"))

    # Build RawData using the canonical function
    raw_data = build_raw_data_from_simple_message(
        message_id=message_id,
        sender=sender,
        content=content,
        timestamp=timestamp,
        sender_name=sender_name,
        role=role,
        group_id=group_id,
        group_name=group_name,
        refer_list=normalized_refer_list,
    )

    # Create and return MemorizeRequest
    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=[raw_data],
        raw_data_type=RawDataType.CONVERSATION,
        user_id_list=[],
        group_id=group_id,
        group_name=group_name,
        current_time=timestamp,
        force_boundary=message_data.get("force_boundary", False),
    )
