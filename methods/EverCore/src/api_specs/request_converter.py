"""
Request converter module

This module contains various functions to convert external request formats to internal Request objects.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional
from datetime import datetime

from api_specs.memory_models import MemoryType, RetrieveMethod
from api_specs.dtos import RetrieveMemRequest, MemorizeRequest, RawData
from api_specs.memory_types import RawDataType, ScenarioType
from api_specs.id_generator import (
    DEFAULT_SESSION_ID,
    generate_single_user_group_id,
    generate_message_id,
    generate_assistant_sender_id,
    validate_input_id,
)
from core.oxm.constants import MAGIC_ALL
from core.observation.logger import get_logger

logger = get_logger(__name__)


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
        # Validate required fields: user_id or group_id at least one is required
        # if not data.get("user_id") and not data.get("group_id"):
        #     raise ValueError("user_id or group_id at least one is required")

        # Handle retrieve_method, use default keyword if not provided

        retrieve_method_str = data.get("retrieve_method", RetrieveMethod.KEYWORD.value)
        logger.debug(f"[DEBUG] retrieve_method_str from data: {retrieve_method_str!r}")

        # Convert string to RetrieveMethod enum
        try:
            retrieve_method = RetrieveMethod(retrieve_method_str)
            logger.debug(f"[DEBUG] converted to: {retrieve_method}")
        except ValueError:
            raise ValueError(
                f"Invalid retrieve_method: {retrieve_method_str}. "
                f"Supported methods: {[m.value for m in RetrieveMethod]}"
            )

        # Convert top_k to integer type (all obtained from query_params are strings)
        # Default to -1 means return all results that meet the threshold
        top_k = data.get("top_k", -1)
        if isinstance(top_k, str):
            top_k = int(top_k)

        # Convert include_metadata to boolean type
        include_metadata = data.get("include_metadata", True)
        if isinstance(include_metadata, str):
            include_metadata = include_metadata.lower() in ("true", "1", "yes")

        # Convert radius to float type (if exists)
        radius = data.get("radius", None)
        if radius is not None and isinstance(radius, str):
            radius = float(radius)

        # Convert memory_types string list to MemoryType enum list
        raw_memory_types = data.get("memory_types", [])
        # Handle comma-separated string (from query_params)
        if isinstance(raw_memory_types, str):
            raw_memory_types = [
                mt.strip() for mt in raw_memory_types.split(",") if mt.strip()
            ]
        memory_types = []
        for mt in raw_memory_types:
            if isinstance(mt, str):
                try:
                    memory_types.append(MemoryType(mt))
                except ValueError:
                    logger.error(f"Invalid memory_type: {mt}, skipping")
            elif isinstance(mt, MemoryType):
                memory_types.append(mt)

        # Default: profile + episodic_memory if not specified
        if not memory_types:
            memory_types = [MemoryType.PROFILE, MemoryType.EPISODIC_MEMORY]

        # Handle group_ids: support both string and array for backward compatibility
        # Priority: group_ids (new) > group_id (old, for backward compatibility)
        group_ids_raw = data.get("group_ids", None)
        if group_ids_raw is None:
            # Try legacy group_id parameter for backward compatibility
            group_id_legacy = data.get("group_id", None)
            if isinstance(group_id_legacy, str):
                group_ids = [group_id_legacy]  # Convert string to array
            elif isinstance(group_id_legacy, list):
                group_ids = group_id_legacy
            else:
                group_ids = None
        elif isinstance(group_ids_raw, str):
            # Support comma-separated string to array (Query Param scenario)
            group_ids = [g.strip() for g in group_ids_raw.split(",") if g.strip()]
            # If parsed result is empty array, set to None
            if not group_ids:
                group_ids = None
        elif isinstance(group_ids_raw, list):
            group_ids = group_ids_raw if group_ids_raw else None
        else:
            group_ids = None

        return RetrieveMemRequest(
            retrieve_method=retrieve_method,
            user_id=data.get(
                "user_id", MAGIC_ALL
            ),  # User ID, use MAGIC_ALL to skip user filtering
            group_ids=group_ids,  # Group IDs array (List[str] or None)
            query=query or data.get("query", None),
            memory_types=memory_types,
            top_k=top_k,
            include_metadata=include_metadata,
            start_time=data.get("start_time", None),
            end_time=data.get("end_time", None),
            radius=radius,  # COSINE similarity threshold
        )
    except Exception as e:
        raise ValueError(f"RetrieveMemRequest conversion failed: {e}")


# =========================================


def _normalize_content_items(content_items: list) -> list:
    """Normalize ContentItem dicts: rename 'content' to 'text' for type='text' items.

    Accepts both legacy {type: "text", content: "..."} and canonical
    {type: "text", text: "..."} formats. Output always uses 'text'.
    """
    normalized = []
    for item in content_items:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        if item.get("type") == "text" and "content" in item and "text" not in item:
            content_value = item["content"]
            item = {k: v for k, v in item.items() if k != "content"}
            item["text"] = content_value
        normalized.append(item)
    return normalized


def build_raw_data_from_message(
    message_id: str,
    sender_id: str,
    content_items: list,
    timestamp: datetime,
    sender_name: Optional[str] = None,
    role: Optional[str] = None,
    tool_calls: Optional[list] = None,
    tool_call_id: Optional[str] = None,
) -> RawData:
    """
    Build RawData object from message fields.

    This is the canonical function for creating RawData from messages.
    The content dict mirrors the v1 API message format, with content as
    a list of content items [{type: "text", text: "..."}].

    Args:
        message_id: Message ID (required)
        sender_id: Sender user ID (required)
        content_items: Content items list [{type, text}] in v1 format (required)
        timestamp: Message timestamp as datetime object (required)
        sender_name: Sender display name (defaults to sender_id if not provided)
        role: Message sender role, "user", "assistant", or "tool" (required)
        tool_calls: Tool calls from assistant (OpenAI format, optional)
        tool_call_id: Tool call ID this message responds to (role=tool, optional)

    Returns:
        RawData: Fully constructed RawData object
    """
    if sender_name is None:
        sender_name = sender_id

    raw_content = {
        "message_id": message_id,
        "sender_id": sender_id,
        "sender_name": sender_name,
        "role": role,
        "content": content_items,
        "timestamp": timestamp,
    }

    # Add OpenAI-format agent fields if present
    if tool_calls:
        raw_content["tool_calls"] = tool_calls
    if tool_call_id:
        raw_content["tool_call_id"] = tool_call_id

    metadata = {"original_id": message_id}

    return RawData(content=raw_content, data_id=message_id, metadata=metadata)


def _unix_ms_to_datetime(unix_ms: int) -> datetime:
    """Convert unix milliseconds timestamp to timezone-aware datetime.

    Uses project timezone from TZ env var (default: UTC).

    Args:
        unix_ms: Unix timestamp in milliseconds

    Returns:
        datetime: Timezone-aware datetime
    """
    from common_utils.datetime_utils import from_timestamp

    return from_timestamp(unix_ms)


def convert_personal_add_to_memorize_request(
    request_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert POST /api/v1/memories (personal add) request to MemorizeRequest.

    Personal add: user_id is the owner. group_id = hash(user_id).
    session_id is propagated through for conversation isolation.

    Args:
        request_data: Personal add request body with fields:
            - user_id (required): Owner user ID
            - session_id (optional): Session identifier
            - messages (required): List of message objects

    Returns:
        MemorizeRequest
    """
    user_id = request_data.get("user_id")
    session_id = request_data.get("session_id", DEFAULT_SESSION_ID)
    messages = request_data.get("messages", [])

    if not user_id:
        raise ValueError("Missing required field: user_id")
    if "session_id" in request_data and request_data["session_id"] is not None:
        validate_input_id("session_id", session_id)
    if not messages:
        raise ValueError("Missing required field: messages")

    # Personal scene: group_id = hash(user_id)
    group_id = generate_single_user_group_id(user_id)

    raw_data_list = []
    latest_timestamp = None

    for msg in messages:
        # Validate content field: accept plain string or array (v1 format)
        content_items = msg.get("content", [])
        if isinstance(content_items, str):
            if not content_items:
                raise ValueError("Missing required field: messages[].content")
            content_items = [{"type": "text", "text": content_items}]
        elif not content_items or not isinstance(content_items, list):
            raise ValueError(
                "Missing required field: messages[].content (must be a non-empty string or array)"
            )
        content_items = _normalize_content_items(content_items)

        # Parse timestamp from unix ms
        created_at_ms = msg.get("timestamp")
        if not created_at_ms:
            raise ValueError("Missing required field: messages[].timestamp")
        timestamp = _unix_ms_to_datetime(created_at_ms)
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp

        user_message_id = msg.get("message_id")
        if user_message_id:
            validate_input_id("message_id", user_message_id)
        message_id = user_message_id or generate_message_id(user_id, created_at_ms)

        role = msg.get("role")
        if not role:
            raise ValueError("Missing required field: messages[].role")
        if role not in ("user", "assistant"):
            raise ValueError(
                f"Invalid value for messages[].role: '{role}'. Must be 'user' or 'assistant'"
            )
        sender_id = msg.get("sender_id")
        if sender_id:
            validate_input_id("sender_id", sender_id)

        # Personal scene: role determines sender_id
        if role == "user":
            if sender_id and sender_id != user_id:
                raise ValueError(
                    f"sender_id mismatch: role=user requires sender_id={user_id}, got {sender_id}"
                )
            sender_id = user_id
        else:
            # role is assistant
            if sender_id and sender_id == user_id:
                raise ValueError(
                    f"sender_id conflict: role={role} cannot use user_id as sender_id"
                )
            if not sender_id:
                sender_id = generate_assistant_sender_id(user_id)

        sender_name = msg.get("sender_name", sender_id)

        raw_data = build_raw_data_from_message(
            message_id=message_id,
            sender_id=sender_id,
            content_items=content_items,
            timestamp=timestamp,
            sender_name=sender_name,
            role=role,
        )
        raw_data_list.append(raw_data)

    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=raw_data_list,
        raw_data_type=RawDataType.CONVERSATION,
        group_id=group_id,
        current_time=latest_timestamp,
        flush=False,
        session_id=session_id,
        scene=ScenarioType.SOLO.value,
    )


def convert_group_add_to_memorize_request(
    request_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert POST /api/v1/memories/group (group add) request to MemorizeRequest.

    Group add: group_id is provided directly. sender_id = user_id internally.

    Args:
        request_data: Group add request body with fields:
            - group_id (required): Group identifier
            - group_meta (optional): Group metadata (name, description)
            - messages (required): List of message objects

    Returns:
        MemorizeRequest
    """
    group_id = request_data.get("group_id")
    messages = request_data.get("messages", [])

    if not group_id:
        raise ValueError("Missing required field: group_id")
    validate_input_id("group_id", group_id)
    if not messages:
        raise ValueError("Missing required field: messages")

    raw_data_list = []
    latest_timestamp = None

    for msg in messages:
        # Validate content field: accept plain string or array (v1 format)
        content_items = msg.get("content", [])
        if isinstance(content_items, str):
            if not content_items:
                raise ValueError("Missing required field: messages[].content")
            content_items = [{"type": "text", "text": content_items}]
        elif not content_items or not isinstance(content_items, list):
            raise ValueError(
                "Missing required field: messages[].content (must be a non-empty string or array)"
            )
        content_items = _normalize_content_items(content_items)

        created_at_ms = msg.get("timestamp")
        if not created_at_ms:
            raise ValueError("Missing required field: messages[].timestamp")
        timestamp = _unix_ms_to_datetime(created_at_ms)
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp

        user_message_id = msg.get("message_id")
        if user_message_id:
            validate_input_id("message_id", user_message_id)
        message_id = user_message_id or generate_message_id(group_id, created_at_ms)

        sender_id = msg.get("sender_id")
        if not sender_id:
            raise ValueError(
                "Missing required field: messages[].sender_id (required for group)"
            )
        validate_input_id("sender_id", sender_id)
        sender_name = msg.get("sender_name", sender_id)
        role = msg.get("role")
        if not role:
            raise ValueError("Missing required field: messages[].role")
        if role not in ("user", "assistant"):
            raise ValueError(
                f"Invalid value for messages[].role: '{role}'. Must be 'user' or 'assistant'"
            )

        raw_data = build_raw_data_from_message(
            message_id=message_id,
            sender_id=sender_id,
            content_items=content_items,
            timestamp=timestamp,
            sender_name=sender_name,
            role=role,
        )
        raw_data_list.append(raw_data)

    # Group scene: no session isolation
    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=raw_data_list,
        raw_data_type=RawDataType.CONVERSATION,
        group_id=group_id,
        current_time=latest_timestamp,
        flush=False,
        session_id=DEFAULT_SESSION_ID,
        scene=ScenarioType.TEAM.value,
    )


def convert_personal_flush_to_memorize_request(
    request_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert POST /api/v1/memories/flush (personal flush) to MemorizeRequest.

    Flush sends no messages, just triggers boundary detection on accumulated data.

    Args:
        request_data: Personal flush request body with fields:
            - user_id (required): Owner user ID
            - session_id (optional): Target session

    Returns:
        MemorizeRequest with empty new_raw_data_list and flush=True
    """
    user_id = request_data.get("user_id")
    session_id = request_data.get("session_id", DEFAULT_SESSION_ID)

    if not user_id:
        raise ValueError("Missing required field: user_id")
    if "session_id" in request_data and request_data["session_id"] is not None:
        validate_input_id("session_id", session_id)

    group_id = generate_single_user_group_id(user_id)

    from common_utils.datetime_utils import get_now_with_timezone

    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=[],
        raw_data_type=RawDataType.CONVERSATION,
        group_id=group_id,
        current_time=get_now_with_timezone(),
        flush=True,
        session_id=session_id,
        scene=ScenarioType.SOLO.value,
    )


def convert_agent_flush_to_memorize_request(
    request_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert POST /api/v1/memories/agent/flush (agent flush) to MemorizeRequest.

    Same as personal flush but with raw_data_type = AGENTCONVERSATION,
    so flush uses AgentMemCellExtractor for boundary detection.

    Args:
        request_data: Agent flush request body with fields:
            - user_id (required): Owner user ID
            - session_id (optional): Target session

    Returns:
        MemorizeRequest with flush=True and raw_data_type=AGENTCONVERSATION
    """
    user_id = request_data.get("user_id")
    session_id = request_data.get("session_id", DEFAULT_SESSION_ID)

    if not user_id:
        raise ValueError("Missing required field: user_id")
    if "session_id" in request_data and request_data["session_id"] is not None:
        validate_input_id("session_id", session_id)

    group_id = generate_single_user_group_id(user_id)

    from common_utils.datetime_utils import get_now_with_timezone

    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=[],
        raw_data_type=RawDataType.AGENTCONVERSATION,
        group_id=group_id,
        current_time=get_now_with_timezone(),
        flush=True,
        session_id=session_id,
        scene=ScenarioType.SOLO.value,
    )


def convert_group_flush_to_memorize_request(
    request_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert POST /api/v1/memories/group/flush (group flush) to MemorizeRequest.

    Flush sends no messages, just triggers boundary detection on accumulated data.

    Args:
        request_data: Group flush request body with fields:
            - group_id (required): Target group

    Returns:
        MemorizeRequest with empty new_raw_data_list and flush=True
    """
    group_id = request_data.get("group_id")

    if not group_id:
        raise ValueError("Missing required field: group_id")
    validate_input_id("group_id", group_id)

    from common_utils.datetime_utils import get_now_with_timezone

    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=[],
        raw_data_type=RawDataType.CONVERSATION,
        group_id=group_id,
        current_time=get_now_with_timezone(),
        flush=True,
        session_id=DEFAULT_SESSION_ID,
        scene=ScenarioType.TEAM.value,
    )


def convert_agent_add_to_memorize_request(
    request_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert POST /api/v1/memories/agent (agent add) request to MemorizeRequest.

    Mirrors personal add logic with agent-specific extensions:
    - role supports "user", "assistant", "tool"
    - tool_calls / tool_call_id stored in RawData content
    - raw_data_type = AGENTCONVERSATION

    Args:
        request_data: Agent add request body with fields:
            - user_id (required): Owner user ID
            - session_id (optional): Session identifier
            - messages (required): List of agent trajectory messages

    Returns:
        MemorizeRequest
    """
    user_id = request_data.get("user_id")
    session_id = request_data.get("session_id", DEFAULT_SESSION_ID)
    messages = request_data.get("messages", [])

    if not user_id:
        raise ValueError("Missing required field: user_id")
    if "session_id" in request_data and request_data["session_id"] is not None:
        validate_input_id("session_id", session_id)
    if not messages:
        raise ValueError("Missing required field: messages")

    # Same as personal: group_id = hash(user_id)
    group_id = generate_single_user_group_id(user_id)

    raw_data_list = []
    latest_timestamp = None

    for msg in messages:
        # Parse timestamp
        created_at_ms = msg.get("timestamp")
        if not created_at_ms:
            raise ValueError("Missing required field: messages[].timestamp")
        timestamp = _unix_ms_to_datetime(created_at_ms)
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp

        # Validate role
        role = msg.get("role")
        if not role:
            raise ValueError("Missing required field: messages[].role")
        if role not in ("user", "assistant", "tool"):
            raise ValueError(
                f"Invalid value for messages[].role: '{role}'. "
                "Must be 'user', 'assistant', or 'tool'"
            )

        # Validate content - accept plain string or array (v1 format)
        content_items = msg.get("content", [])
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        if role == "tool" and not tool_call_id:
            raise ValueError(
                "Missing required field: messages[].tool_call_id (required when role='tool')"
            )

        # assistant messages with tool_calls may have empty/null content
        if isinstance(content_items, str):
            if not content_items:
                if role == "assistant" and tool_calls:
                    content_items = [{"type": "text", "text": ""}]
                else:
                    raise ValueError("Missing required field: messages[].content")
            else:
                content_items = [{"type": "text", "text": content_items}]
        elif not content_items or not isinstance(content_items, list):
            if role == "assistant" and tool_calls:
                content_items = [{"type": "text", "text": ""}]
            else:
                raise ValueError(
                    "Missing required field: messages[].content (must be a non-empty string or array)"
                )
        content_items = _normalize_content_items(content_items)

        # Message ID
        user_message_id = msg.get("message_id")
        if user_message_id:
            validate_input_id("message_id", user_message_id)
        message_id = user_message_id or generate_message_id(user_id, created_at_ms)

        # sender_id logic
        sender_id = msg.get("sender_id")
        if role == "user":
            if sender_id and sender_id != user_id:
                raise ValueError(
                    f"sender_id mismatch: role=user requires sender_id={user_id}, got {sender_id}"
                )
            sender_id = user_id
        elif role == "assistant":
            if sender_id and sender_id == user_id:
                raise ValueError(
                    f"sender_id conflict: role=assistant cannot use user_id as sender_id"
                )
            if not sender_id:
                hash_val = hashlib.md5(f"{user_id}_assistant".encode()).hexdigest()[:12]
                sender_id = f"{hash_val}_assistant"
        elif role == "tool":
            if not sender_id:
                hash_val = hashlib.md5(
                    f"{user_id}_tool_{tool_call_id}".encode()
                ).hexdigest()[:12]
                sender_id = f"{hash_val}_tool"

        sender_name = msg.get("sender_name", sender_id)

        raw_data = build_raw_data_from_message(
            message_id=message_id,
            sender_id=sender_id,
            content_items=content_items,
            timestamp=timestamp,
            sender_name=sender_name,
            role=role,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        raw_data_list.append(raw_data)

    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=raw_data_list,
        raw_data_type=RawDataType.AGENTCONVERSATION,
        group_id=group_id,
        current_time=latest_timestamp,
        flush=False,
        session_id=session_id,
        scene=ScenarioType.SOLO.value,
    )
