"""Memory resource DTOs.

This module contains DTOs related to memory CRUD operations:
- Add / Flush (POST /api/v1/memories, /api/v1/memories/group, etc.)
- Search (GET /api/v1/memories/search)
- Delete (DELETE /api/v1/memories)
- Get (POST /api/v1/memories/get)
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import re

from api_specs.memory_types import ScenarioType
from bson import ObjectId
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    SkipValidation,
    SerializeAsAny,
)

from api_specs.dtos.base import BaseApiResponse
from api_specs.memory_types import RetrieveMemoryModel, RawDataType
from api_specs.memory_models import (
    MemoryType,
    Metadata,
    QueryMetadata,
    RetrieveMethod,
    MessageSenderRole,
)
from core.oxm.constants import MAGIC_ALL, MAX_RETRIEVE_LIMIT
from biz_layer.retrieve_constants import MAX_GROUP_IDS_COUNT


iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'


# =============================================================================
# Raw Data Types
# =============================================================================


@dataclass
class RawData:
    """Raw data structure for storing original content.

    This is oriented towards input at a higher level; the one in the memcell
    table is the storage structure, which is more low-level.
    """

    content: dict[str, Any]
    data_id: str
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Note: Enrichment results (parsed_content, parsed_summary, parse_status)
    # are embedded directly into content item dicts by the enrich provider,
    # so they are persisted as part of content_items in RawMessage.

    def _serialize_value(self, value: Any) -> Any:
        """
        Recursively serialize values, handling special types like datetime and ObjectId

        Args:
            value: Value to be serialized

        Returns:
            JSON-serializable value
        """
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, ObjectId):
            # Serialize ObjectId to string
            return str(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif hasattr(value, '__dict__'):
            # Handle custom objects by converting to dictionary
            return self._serialize_value(value.__dict__)
        else:
            return value

    def _deserialize_value(self, value: Any, field_name: str = "") -> Any:
        """
        Recursively deserialize values, heuristically determining whether to restore datetime type based on field name

        Args:
            value: Value to be deserialized
            field_name: Field name, used for heuristic judgment

        Returns:
            Deserialized value
        """
        if isinstance(value, str):
            # Heuristically determine if it's a datetime field based on field name
            if self._is_datetime_field(field_name) and self._is_iso_datetime(value):
                try:
                    from common_utils.datetime_utils import from_iso_format

                    return from_iso_format(value)
                except (ValueError, ImportError):
                    return value
            return value
        elif isinstance(value, dict):
            return {k: self._deserialize_value(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._deserialize_value(item, field_name) for item in value]
        else:
            return value

    def _is_datetime_field(self, field_name: str) -> bool:
        """
        Heuristically determine if a field is a datetime field based on its name

        Args:
            field_name: Field name

        Returns:
            bool: Whether the field is a datetime field
        """
        if not isinstance(field_name, str):
            return False

        # Exact match datetime field names (based on actual field names used in the project)
        exact_datetime_fields = {
            'timestamp',
            'createTime',
            'updateTime',
            'create_time',
            'update_time',
            'sent_timestamp',
            'received_timestamp',
            'create_timestamp',
            'last_update_timestamp',
            'modify_timestamp',
            'created_at',
            'updated_at',
            'joinTime',
            'leaveTime',
            'lastOnlineTime',
            'sync_time',
            'processed_at',
            'start_time',
            'end_time',
            'event_time',
            'build_timestamp',
            'datetime',
            'created',
            'updated',  # Add common datetime field variants
        }

        field_lower = field_name.lower()

        # Exact match check
        if field_name in exact_datetime_fields or field_lower in exact_datetime_fields:
            return True

        # Exclude common words that should not be recognized as datetime fields
        exclusions = {
            'runtime',
            'timeout',
            'timeline',
            'timestamp_format',
            'time_zone',
            'time_limit',
            'timestamp_count',
            'timestamp_enabled',
            'time_sync',
            'playtime',
            'lifetime',
            'uptime',
            'downtime',
        }

        if field_name in exclusions or field_lower in exclusions:
            return False

        # Suffix match check (stricter rules)
        time_suffixes = ['_time', '_timestamp', '_at', '_date']
        for suffix in time_suffixes:
            if field_name.endswith(suffix) or field_lower.endswith(suffix):
                return True

        # Prefix match check (stricter rules)
        if field_name.endswith('Time') and not field_name.endswith('runtime'):
            # Match xxxTime pattern, but exclude runtime
            return True

        if field_name.endswith('Timestamp'):
            # Match xxxTimestamp pattern
            return True

        return False

    def _is_iso_datetime(self, value: str) -> bool:
        """
        Check if string is ISO format datetime

        Args:
            value: String value

        Returns:
            bool: Whether it is ISO datetime format
        """
        # Simple ISO datetime format check
        if not isinstance(value, str) or len(value) < 19:
            return False

        # Check basic ISO format pattern: YYYY-MM-DDTHH:MM:SS
        return bool(re.match(iso_pattern, value))

    def to_json(self) -> str:
        """
        Serialize RawData object to JSON string

        Returns:
            str: JSON string
        """
        try:
            data = {
                'content': self._serialize_value(self.content),
                'data_id': self.data_id,
                'data_type': self.data_type,
                'metadata': (
                    self._serialize_value(self.metadata) if self.metadata else None
                ),
            }
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize RawData to JSON: {e}") from e

    @classmethod
    def from_json_str(cls, json_str: str) -> 'RawData':
        """
        Deserialize RawData object from JSON string

        Args:
            json_str: JSON string

        Returns:
            RawData: Deserialized RawData object

        Raises:
            ValueError: JSON format error or missing required fields
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON format error: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("JSON must be an object")

        # Check required fields
        if 'content' not in data or 'data_id' not in data:
            raise ValueError("JSON missing required fields: content and data_id")

        # Create instance and deserialize values
        instance = cls.__new__(cls)
        instance.content = instance._deserialize_value(data['content'], 'content')
        instance.data_id = data['data_id']
        instance.data_type = data.get('data_type')
        instance.metadata = (
            instance._deserialize_value(data.get('metadata'), 'metadata')
            if data.get('metadata')
            else None
        )

        return instance


# =============================================================================
# Memorize Internal DTO
# =============================================================================


class MemorizeRequest(BaseModel):
    """Memory storage request (internal business layer)"""

    history_raw_data_list: list[RawData]
    new_raw_data_list: list[RawData]
    raw_data_type: RawDataType
    group_id: Optional[str] = None
    current_time: Optional[datetime] = None
    # Session identifier for conversation isolation
    session_id: Optional[str] = None
    # Optional extraction control parameters
    enable_foresight_extraction: bool = True  # Whether to extract foresight
    enable_atomic_fact_extraction: bool = True  # Whether to extract atomic facts
    # Force boundary trigger - when True, immediately triggers memory extraction
    flush: bool = False
    # Scene type: "solo" (1 user + N agents) or "team" (multi-user + agents)
    scene: str = ScenarioType.SOLO.value

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Add / Flush DTOs
# =============================================================================


class ContentItem(BaseModel):
    """Single content item in a message's content array.

    Supports multimodal content types. This phase only supports type="text".
    Non-text types (audio, image, doc, pdf, html, email) are planned for next phase.
    """

    type: str = Field(
        ...,
        description='Content type: "text" / "audio" / "image" / "doc" / "pdf" / "html" / "email"',
    )
    text: Optional[str] = Field(
        default=None,
        description="Content body. For type='text', this is the actual text. "
        "For other types (image, audio, etc.), this can be a textual description.",
    )
    source: Optional[str] = Field(
        default=None,
        description='Content source: "google_doc" / "notion" / "confluence" / "zoom"',
    )
    base64: Optional[str] = Field(default=None, description="Base64-encoded content")
    uri: Optional[str] = Field(default=None, description="File URI (MinIO, HTTP, etc.)")
    ext: Optional[str] = Field(
        default=None, description="File extension (e.g., 'png', 'mp3', 'pdf')"
    )
    name: Optional[str] = Field(default=None, description="File name")
    source_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Source-related traceability info"
    )
    extras: Optional[Dict[str, Any]] = Field(
        default=None, description="Type-specific extra fields"
    )
    # Note: Enrichment results (parsed_content, parsed_summary, parse_status) are
    # embedded directly into content-item dicts by the enrich provider.

    @model_validator(mode="before")
    @classmethod
    def _compat_content_field(cls, values: Any) -> Any:
        """Accept legacy 'content' field as an alias for 'text'."""
        if isinstance(values, dict) and "content" in values and "text" not in values:
            values = dict(values)
            values["text"] = values.pop("content")
        return values


class MessageItem(BaseModel):
    """Single message item in add request.

    Uses content array for multimodal support. This phase only supports type='text' items.
    content accepts a plain string shorthand: "hello" is coerced to
    [{"type": "text", "text": "hello"}].
    """

    message_id: Optional[str] = Field(default=None, description="Message unique ID")
    sender_id: Optional[str] = Field(default=None, description="Sender identifier")
    sender_name: Optional[str] = Field(default=None, description="Sender display name")
    role: str = Field(..., description="user / assistant")
    timestamp: int = Field(..., description="Message timestamp in unix milliseconds")
    content: Union[str, List[ContentItem]] = Field(
        ...,
        min_length=1,
        description='Content items. Accepts plain string shorthand: "hello" → [{type: "text", text: "hello"}]',
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_content_string(cls, values: Any) -> Any:
        """Coerce plain string content to ContentItem array."""
        if isinstance(values, dict) and isinstance(values.get("content"), str):
            text = values["content"]
            if not text:
                raise ValueError("messages[].content must not be empty")
            values = dict(values)
            values["content"] = [{"type": "text", "text": text}]
        return values


class PersonalAddRequest(BaseModel):
    """POST /api/v1/memories (personal add) request body."""

    user_id: str = Field(..., description="Owner user ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    messages: List[MessageItem] = Field(
        ..., min_length=1, max_length=500, description="Batch message array"
    )


class GroupAddRequest(BaseModel):
    """POST /api/v1/memories/group (group add) request body."""

    group_id: str = Field(..., description="Group identifier")
    group_meta: Optional[Dict[str, Any]] = Field(
        default=None, description="Group metadata"
    )
    messages: List[MessageItem] = Field(
        ..., min_length=1, max_length=500, description="Batch message array"
    )

    @model_validator(mode="after")
    def validate_sender_id_required(self) -> "GroupAddRequest":
        """Validate that sender_id is required for each message in group add."""
        for i, msg in enumerate(self.messages):
            if not msg.sender_id:
                raise ValueError(f"messages[{i}].sender_id is required for group add")
        return self


# ==================== Agent Add (POST /api/v1/memories/agent) ====================


class ToolCallFunction(BaseModel):
    """Function details within a tool call."""

    name: str = Field(..., description="Function/tool name")
    arguments: str = Field(..., description="JSON-encoded arguments string")


class ToolCall(BaseModel):
    """OpenAI-format tool call made by the assistant."""

    id: str = Field(..., description="Unique tool call ID")
    type: str = Field(default="function", description="Tool call type")
    function: ToolCallFunction = Field(..., description="Function call details")


class AgentMessageItem(MessageItem):
    """Extended MessageItem with agent-specific fields.

    Supports role='tool' in addition to 'user'/'assistant'.
    Adds tool_calls (assistant) and tool_call_id (tool) fields.

    Overrides content to Optional: assistant messages with tool_calls may have
    empty/null content (common in OpenAI Chat Completion API). Fine-grained
    validation (user/tool must have content) is handled in the request converter.
    """

    role: str = Field(
        ...,
        description="Message sender role: 'user', 'assistant', or 'tool'",
        examples=["user", "assistant", "tool"],
    )
    content: Optional[Union[str, List[ContentItem]]] = Field(
        default=None,
        description="Content items. Optional for assistant messages with tool_calls.",
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="Tool calls made by the assistant (OpenAI format). "
        "Only applicable when role='assistant'.",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="ID of the tool call this message is responding to. "
        "Required when role='tool'.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_content_string(cls, values: Any) -> Any:
        """Override parent: allow empty content for assistant with tool_calls."""
        if isinstance(values, dict) and isinstance(values.get("content"), str):
            text = values["content"]
            if not text:
                if values.get("role") == "assistant" and values.get("tool_calls"):
                    values = dict(values)
                    values["content"] = [{"type": "text", "text": ""}]
                    return values
                raise ValueError("messages[].content must not be empty")
            values = dict(values)
            values["content"] = [{"type": "text", "text": text}]
        return values


class AgentAddRequest(BaseModel):
    """POST /api/v1/memories/agent request body.

    Strictly mirrors PersonalAddRequest structure, with AgentMessageItem
    supporting tool_calls/tool_call_id and role='tool'.
    """

    user_id: str = Field(..., description="Owner user ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    messages: List[AgentMessageItem] = Field(
        ..., min_length=1, max_length=500, description="Agent trajectory messages"
    )


# ==================== Flush ====================


class PersonalFlushRequest(BaseModel):
    """POST /api/v1/memories/flush (personal flush) request body."""

    user_id: str = Field(..., description="Owner user ID")
    session_id: Optional[str] = Field(default=None, description="Target session")


class GroupFlushRequest(BaseModel):
    """POST /api/v1/memories/group/flush (group flush) request body."""

    group_id: str = Field(..., description="Target group")


class AgentFlushRequest(BaseModel):
    """POST /api/v1/memories/agent/flush (agent flush) request body."""

    user_id: str = Field(..., description="Owner user ID")
    session_id: Optional[str] = Field(default=None, description="Target session")


class AddResult(BaseModel):
    """Add endpoint result data."""

    request_id: str = Field(default="", description="Request tracking ID (reserved)")
    message_count: int = Field(
        default=0, description="Number of messages accepted in this request"
    )
    status: str = Field(
        default="accumulated",
        description="Processing status. "
        "'accumulated': messages buffered, waiting for boundary detection; "
        "'extracted': boundary detected and memory extraction triggered",
        examples=["accumulated", "extracted"],
    )
    message: str = Field(
        default="Messages accepted", description="Human-readable status description"
    )


class AddResponse(BaseApiResponse[AddResult]):
    """Add endpoint response."""

    data: AddResult = Field(default_factory=AddResult, description="Add result")


class FlushResult(BaseModel):
    """Flush endpoint result data."""

    request_id: str = Field(default="", description="Request tracking ID (reserved)")
    status: str = Field(
        default="no_extraction",
        description="Processing status. "
        "'extracted': boundary detected and memory extraction triggered; "
        "'no_extraction': no accumulated messages or no boundary detected",
        examples=["extracted", "no_extraction"],
    )
    message: str = Field(
        default="Flush completed", description="Human-readable status description"
    )


class FlushResponse(BaseApiResponse[FlushResult]):
    """Flush endpoint response."""

    data: FlushResult = Field(default_factory=FlushResult, description="Flush result")


# =============================================================================
# Search/Retrieve DTOs (GET /api/v1/memories/search)
# =============================================================================


class RetrieveMemRequest(BaseModel):
    """Memory retrieve/search request

    Used for GET /api/v1/memories/search endpoint.
    Supports passing parameters via query params or body.
    """

    user_id: Optional[str] = Field(
        default=None,
        description="User ID (at least one of user_id or group_id must be provided)",
        examples=["user_123"],
    )
    group_ids: Optional[List[str]] = Field(
        default=None,
        description="Array of Group IDs to search (max 10 items). "
        "None means search all groups for the user.",
        examples=[["group_456", "group_789"]],
    )
    memory_types: List[MemoryType] = Field(
        default_factory=list,
        description="""List of memory types to retrieve, enum values from MemoryType:
- profile: user profile (Milvus vector search only)
- episodic_memory: episodic memory
- foresight: prospective memory (not yet supported for search)
- atomic_fact: atomic fact (not yet supported for search)
Note: Only profile and episodic_memory are supported. Defaults to both if not specified.""",
        examples=[[MemoryType.EPISODIC_MEMORY.value]],
    )
    top_k: int = Field(
        default=-1,
        description="Maximum number of results to return. -1 means return all results that meet the threshold (up to 100). Valid values: -1 or 1-100.",
        ge=-1,
        le=100,
        examples=[10, -1],
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata", examples=[True]
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Time range start (ISO 8601 format). Only applies to episodic_memory, ignored for profile",
        examples=["2024-01-01T00:00:00"],
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Time range end (ISO 8601 format). Only applies to episodic_memory, ignored for profile",
        examples=["2024-12-31T23:59:59"],
    )
    query: Optional[str] = Field(
        default=None, description="Search query text", examples=["coffee preference"]
    )
    retrieve_method: RetrieveMethod = Field(
        default=RetrieveMethod.KEYWORD,
        description="""Retrieval method, enum values from RetrieveMethod:
- keyword: keyword retrieval (BM25, default)
- vector: vector semantic retrieval
- hybrid: hybrid retrieval (keyword + vector)
- rrf: RRF fusion retrieval (keyword + vector + RRF ranking fusion)
- agentic: LLM-guided multi-round intelligent retrieval""",
        examples=["keyword"],
    )
    radius: Optional[float] = Field(
        default=None,
        description="COSINE similarity threshold for vector retrieval (only for vector and hybrid methods, default 0.6)",
        ge=0.0,
        le=1.0,
        examples=[0.6],
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_request(self) -> "RetrieveMemRequest":
        """Validate request parameters"""
        # Validate: at least one of user_id or group_ids must be specified
        if (
            self.user_id is None or self.user_id == MAGIC_ALL
        ) and self.group_ids is None:
            raise ValueError(
                "At least one of user_id or group_ids must be specified. "
                "Cannot query without any filter."
            )

        # Validate: user_id is not specified and group_ids is an empty list
        if (
            (self.user_id is None or self.user_id == MAGIC_ALL)
            and isinstance(self.group_ids, list)
            and len(self.group_ids) == 0
        ):
            raise ValueError(
                "group_ids cannot be an empty list when user_id is not specified."
            )

        # Validate: group_ids array length cannot exceed MAX_GROUP_IDS_COUNT
        if self.group_ids is not None and len(self.group_ids) > MAX_GROUP_IDS_COUNT:
            raise ValueError(
                f"group_ids array length cannot exceed {MAX_GROUP_IDS_COUNT}"
            )

        # Validate: Search supports episodic_memory, profile, agent_case, agent_skill
        if self.memory_types:
            allowed_types = {
                MemoryType.EPISODIC_MEMORY,
                MemoryType.PROFILE,
                MemoryType.AGENT_CASE,
                MemoryType.AGENT_SKILL,
            }
            invalid_types = [mt for mt in self.memory_types if mt not in allowed_types]
            if invalid_types:
                raise ValueError(
                    f"Search interface only supports memory_types: "
                    f"episodic_memory, profile, agent_case, agent_skill. "
                    f"Invalid types: {[mt.value for mt in invalid_types]}"
                )

        # top_k must be -1 (return all) or positive (1-100), 0 is invalid
        if self.top_k == 0:
            raise ValueError(
                "top_k must be -1 (return all results) or a positive integer (1-100)"
            )

        if self.top_k > 0 and self.top_k > MAX_RETRIEVE_LIMIT:
            object.__setattr__(self, "top_k", MAX_RETRIEVE_LIMIT)

        return self


class RawMessageDTO(BaseModel):
    """Raw message DTO for messages not yet extracted into memory.

    Represents a cached message waiting for boundary detection or memory extraction.
    """

    id: str  # MongoDB ObjectId as string
    request_id: str  # Request ID
    message_id: Optional[str] = None  # Message ID
    group_id: Optional[str] = None  # Group ID
    session_id: Optional[str] = None  # Session identifier for conversation isolation
    sender_id: Optional[str] = None  # Sender ID
    sender_name: Optional[str] = None  # Sender name
    content_items: Optional[List[Dict[str, Any]]] = None  # Message content items list
    timestamp: Optional[str] = None  # Message timestamp (ISO 8601 format with timezone)
    created_at: Optional[str] = None  # Record creation time (ISO 8601 format)
    updated_at: Optional[str] = None  # Record update time (ISO 8601 format)


class ProfileSearchItem(BaseModel):
    """Profile search result item.

    Represents a single profile item from Milvus vector search.
    Fields are parsed from embed_text.
    """

    item_type: str = Field(
        description="Item type: explicit_info or implicit_trait",
        examples=["explicit_info", "implicit_trait"],
    )
    # For explicit_info
    category: Optional[str] = Field(
        default=None,
        description="Category name (for explicit_info type)",
        examples=["Dietary Preferences", "Professional Skills"],
    )
    # For implicit_trait
    trait_name: Optional[str] = Field(
        default=None,
        description="Trait name (for implicit_trait type)",
        examples=["Health Conscious", "Efficiency Focused"],
    )
    description: str = Field(
        default="",
        description="Description content",
        examples=["Prefers light flavors, favoring vegetables and seafood."],
    )
    score: float = Field(
        default=0.0,
        description="Similarity score from Milvus search",
        examples=[0.89, 0.75],
    )


class RetrieveMemResponse(BaseModel):
    """Memory retrieve/search response (result data) - flat structure"""

    # Profile search results (from Milvus, no rerank)
    profiles: List[ProfileSearchItem] = Field(
        default_factory=list,
        description="Profile search results (explicit_info and implicit_traits)",
    )
    memories: SkipValidation[List[RetrieveMemoryModel]] = Field(default_factory=list)
    total_count: int = 0
    query_metadata: SkipValidation[Optional[QueryMetadata]] = None
    metadata: SkipValidation[Optional[Metadata]] = None
    pending_messages: SkipValidation[List[RawMessageDTO]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class SearchMemoriesResponse(BaseApiResponse[RetrieveMemResponse]):
    """Memory search API response

    Response for GET /api/v1/memories/search endpoint.
    """

    data: RetrieveMemResponse = Field(description="Memory search result")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "Memory search successful",
                "result": {
                    "profiles": [
                        {
                            "item_type": "explicit_info",
                            "category": "Dietary Preferences",
                            "description": "Prefers light flavors, favoring vegetables and seafood",
                            "score": 0.89,
                        },
                        {
                            "item_type": "implicit_trait",
                            "trait_name": "Health Conscious",
                            "description": "Prioritizes dietary health, preferring low oil and low salt",
                            "score": 0.75,
                        },
                    ],
                    "memories": [
                        {
                            "memory_type": "episodic_memory",
                            "user_id": "user_123",
                            "timestamp": "2024-01-15T10:30:00",
                            "summary": "User mentioned controlling their diet recently, eating only two meals a day, with dinner mainly being salad",
                            "group_id": "group_456",
                        }
                    ],
                    "scores": [0.82],
                    "original_data": [],
                    "total_count": 3,
                    "has_more": False,
                    "query_metadata": {
                        "source": "hybrid_search",
                        "user_id": "user_123",
                        "memory_type": "retrieve",
                    },
                    "metadata": {
                        "profile_count": 2,
                        "episodic_count": 1,
                        "latency_ms": 156,
                    },
                    "pending_messages": [],
                },
            }
        }
    }


# =============================================================================
# Delete DTOs (DELETE /api/v1/memories)
# =============================================================================


class DeleteMemoriesRequest(BaseModel):
    """
    Delete memories request body

    Used for DELETE /api/v1/memories endpoint

    Notes:
    - memory_id, user_id, group_id are combined filter conditions
    - If all three are provided, all conditions must be met
    - If not provided, use MAGIC_ALL ("__all__") to skip filtering
    - Cannot all be MAGIC_ALL (at least one filter required)
    - id and event_id are aliases for memory_id (backward compatibility)
    """

    memory_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="Memory id (filter condition)",
        examples=["507f1f77bcf86cd799439011", MAGIC_ALL],
    )
    # Backward compatibility: support id and event_id as alias for memory_id
    id: Optional[str] = Field(
        default=None,
        description="Alias for memory_id (backward compatibility)",
        examples=["507f1f77bcf86cd799439011"],
    )
    event_id: Optional[str] = Field(
        default=None,
        description="Alias for memory_id (backward compatibility)",
        examples=["507f1f77bcf86cd799439011"],
    )
    user_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="User ID (filter condition)",
        examples=["user_123", MAGIC_ALL],
    )
    group_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="Group ID (filter condition)",
        examples=["group_456", MAGIC_ALL],
    )

    @model_validator(mode="after")
    def validate_filters(self):
        """Validate that at least one filter is provided"""
        # Resolve memory_id from aliases (priority: memory_id > id > event_id)
        effective_memory_id = self.memory_id
        if effective_memory_id == MAGIC_ALL:
            effective_memory_id = self.id or self.event_id or MAGIC_ALL

        # Check if all are MAGIC_ALL
        if (
            effective_memory_id == MAGIC_ALL
            and self.user_id == MAGIC_ALL
            and self.group_id == MAGIC_ALL
        ):
            raise ValueError(
                "At least one of memory_id, user_id, or group_id must be provided (not MAGIC_ALL)"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Delete by memory_id only",
                    "value": {
                        "memory_id": "507f1f77bcf86cd799439011",
                        "user_id": MAGIC_ALL,
                        "group_id": MAGIC_ALL,
                    },
                },
                {
                    "summary": "Delete by user_id only",
                    "value": {
                        "memory_id": MAGIC_ALL,
                        "user_id": "user_123",
                        "group_id": MAGIC_ALL,
                    },
                },
                {
                    "summary": "Delete by user_id and group_id",
                    "value": {
                        "memory_id": MAGIC_ALL,
                        "user_id": "user_123",
                        "group_id": "group_456",
                    },
                },
            ]
        }
    }


class DeleteMemoriesResult(BaseModel):
    """Delete memories result data"""

    filters: List[str] = Field(
        default_factory=list,
        description="List of filter types used for deletion",
        examples=[["event_id"], ["user_id", "group_id"]],
    )
    count: int = Field(
        default=0, description="Number of memories deleted", examples=[1, 25]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Delete by event_id only",
                    "value": {"filters": ["event_id"], "count": 1},
                },
                {
                    "summary": "Delete by user_id only",
                    "value": {"filters": ["user_id"], "count": 25},
                },
                {
                    "summary": "Delete by user_id and group_id",
                    "value": {"filters": ["user_id", "group_id"], "count": 10},
                },
            ]
        }
    }


class DeleteMemoriesResponse(BaseApiResponse[DeleteMemoriesResult]):
    """Delete memories API response

    Response for DELETE /api/v1/memories endpoint.
    """

    data: DeleteMemoriesResult = Field(description="Delete operation result")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Delete by event_id only",
                    "value": {
                        "status": "ok",
                        "message": "Successfully deleted 1 memory",
                        "result": {"filters": ["event_id"], "count": 1},
                    },
                },
                {
                    "summary": "Delete by user_id only",
                    "value": {
                        "status": "ok",
                        "message": "Successfully deleted 25 memories",
                        "result": {"filters": ["user_id"], "count": 25},
                    },
                },
                {
                    "summary": "Delete by user_id and group_id",
                    "value": {
                        "status": "ok",
                        "message": "Successfully deleted 10 memories",
                        "result": {"filters": ["user_id", "group_id"], "count": 10},
                    },
                },
            ]
        }
    }


# =============================================================================
# Get DTOs (POST /api/v1/memories/get)
# =============================================================================


class GetMemRequest(BaseModel):
    """Memory get request

    Used for POST /api/v1/memories/get endpoint.

    Note:
    - memory_type: supported values are "episodic_memory", "profile", "agent_case", "agent_skill"
    - filters must contain at least one of user_id or group_id at first level
    - filters supports operators: eq (implicit), in, gt, gte, lt, lte
    - filters supports combinators: AND, OR
    """

    memory_type: str = Field(
        description="Memory type to get: episodic_memory, profile, agent_case, agent_skill",
        examples=[MemoryType.EPISODIC_MEMORY.value],
    )
    page: int = Field(
        default=1, description="Page number, starts from 1", ge=1, examples=[1]
    )
    page_size: int = Field(
        default=20,
        description="Items per page, default 20, max 100",
        ge=1,
        le=100,
        examples=[20],
    )
    rank_by: str = Field(
        default="timestamp", description="Sort field", examples=["timestamp"]
    )
    rank_order: str = Field(
        default="desc", description="Sort order: asc or desc", examples=["desc"]
    )
    filters: Dict[str, Any] = Field(
        description="Filter conditions with user_id/group_id scope and optional operators"
    )

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, v: str) -> str:
        allowed = {
            MemoryType.EPISODIC_MEMORY.value,
            MemoryType.PROFILE.value,
            MemoryType.AGENT_CASE.value,
            MemoryType.AGENT_SKILL.value,
        }
        if v not in allowed:
            raise ValueError(
                f"memory_type must be one of: {', '.join(sorted(allowed))}"
            )
        return v

    @field_validator("rank_order")
    @classmethod
    def validate_rank_order(cls, v: str) -> str:
        if v not in ("asc", "desc"):
            raise ValueError("rank_order must be 'asc' or 'desc'")
        return v

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if "user_id" not in v and "group_id" not in v:
            raise ValueError(
                "filters must contain at least one of 'user_id' or 'group_id' at first level"
            )
        return v


class EpisodeItem(BaseModel):
    """Episode object in GET response

    Derived from episodic_memories collection.
    12 fields, no score, no vector, no audit timestamps.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(
        default=None, description="Owner user ID, null = group memory"
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: Optional[datetime] = Field(
        default=None, description="Event occurrence time"
    )
    participants: Optional[List[str]] = Field(
        default=None, description="Event participant names"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of event participants"
    )
    summary: Optional[str] = Field(default=None, description="Memory summary")
    subject: Optional[str] = Field(default=None, description="Memory subject")
    episode: Optional[str] = Field(
        default=None, description="Full episodic memory text"
    )
    type: Optional[str] = Field(default=None, description="Episode type")
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")


class ProfileItem(BaseModel):
    """Profile object in GET response

    Derived from user_profiles collection.
    6 fields, no audit timestamps.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    profile_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Profile data"
    )
    scenario: Optional[str] = Field(
        default=None, description="Scenario type: solo or team"
    )
    memcell_count: Optional[int] = Field(default=None, description="Number of MemCells")


class GetMemResponse(BaseModel):
    """Memory get response data

    Response for POST /api/v1/memories/get endpoint.
    Wrapped in envelope: { "data": GetMemResponse }
    """

    episodes: List[EpisodeItem] = Field(
        default_factory=list, description="Episodic memory items"
    )
    profiles: List[ProfileItem] = Field(
        default_factory=list, description="Profile items"
    )
    agent_cases: List[AgentCaseItem] = Field(
        default_factory=list,
        description="Agent case items (populated when memory_type=agent_case)",
    )
    agent_skills: List[AgentSkillItem] = Field(
        default_factory=list,
        description="Agent skill items (populated when memory_type=agent_skill)",
    )
    total_count: int = Field(
        default=0, description="Total number of records matching query conditions"
    )
    count: int = Field(default=0, description="Number of records in current page")


class GetMemoriesResponse(BaseModel):
    """Response envelope for POST /api/v1/memories/get

    Used as response_model for OpenAPI documentation.
    """

    data: GetMemResponse = Field(description="Memory get result")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {
                        "episodes": [
                            {
                                "id": "67c8a1b2f3e4d5c6a7b8c9d0",
                                "user_id": "user_123",
                                "group_id": "group_abc",
                                "session_id": "sess_abc",
                                "timestamp": "2026-03-01T10:00:00Z",
                                "participants": ["user_123", "user_456"],
                                "summary": "Team discussed Q1 roadmap",
                                "subject": "Q1 Roadmap Discussion",
                                "episode": "Alice and Bob discussed...",
                                "type": "Conversation",
                                "parent_type": "memcell",
                                "parent_id": "67c8a1b2f3e4d5c6a7b8c9d1",
                            }
                        ],
                        "profiles": [],
                        "agent_cases": [],
                        "agent_skills": [],
                        "total_count": 1,
                        "count": 1,
                    }
                },
                {
                    "data": {
                        "episodes": [],
                        "profiles": [
                            {
                                "id": "67c8a1b2f3e4d5c6a7b8c9e0",
                                "user_id": "user_123",
                                "group_id": "group_abc",
                                "profile_data": {
                                    "explicit_info": {"Role": "Product Manager"},
                                    "implicit_traits": {
                                        "Leadership": "Takes initiative in meetings"
                                    },
                                },
                                "scenario": "team",
                                "memcell_count": 5,
                            }
                        ],
                        "agent_cases": [],
                        "agent_skills": [],
                        "total_count": 1,
                        "count": 1,
                    }
                },
                {
                    "data": {
                        "episodes": [],
                        "profiles": [],
                        "agent_cases": [
                            {
                                "id": "67d1a2b3c4e5f6a7b8c9d0e1",
                                "user_id": "user_01",
                                "group_id": None,
                                "session_id": "sess_agent_001",
                                "task_intent": "Retrieve and summarize weather data for a given city",
                                "approach": "1. Parse city name from user query. 2. Call get_weather API. 3. Format response with temperature and conditions.",
                                "quality_score": 0.92,
                                "timestamp": "2026-03-15T14:30:00Z",
                                "parent_type": "memcell",
                                "parent_id": "67d1a2b3c4e5f6a7b8c9d0e2",
                            }
                        ],
                        "agent_skills": [],
                        "total_count": 1,
                        "count": 1,
                    }
                },
                {
                    "data": {
                        "episodes": [],
                        "profiles": [],
                        "agent_cases": [],
                        "agent_skills": [
                            {
                                "id": "67d2b3c4d5e6f7a8b9c0d1e2",
                                "user_id": "user_01",
                                "group_id": None,
                                "cluster_id": "cluster_weather_01",
                                "name": "Weather Query Handling",
                                "description": "Retrieve weather data for cities using the get_weather API and present results in a user-friendly format",
                                "content": "Steps: 1. Extract city name. 2. Call get_weather(city). 3. Format: '{city}: {temp}, {conditions}'.",
                                "confidence": 0.88,
                                "maturity_score": 0.75,
                            }
                        ],
                        "total_count": 1,
                        "count": 1,
                    }
                },
            ]
        }
    }


# =============================================================================
# Search DTOs (POST /api/v1/memories/search)
# =============================================================================


class SearchMemoriesRequest(BaseModel):
    """Memory search request (v1)

    Used for POST /api/v1/memories/search endpoint.
    Uses structured Filters DSL (MongoFilterParser compatible).

    Note:
    - query: Search query text (required for keyword/vector/hybrid/rrf/agentic methods)
    - method: Retrieval method (keyword/vector/hybrid/rrf/agentic, default: hybrid)
    - memory_types: List of memory types to search (episodic_memory, profile, raw_message, agent_memory)
    - filters: Structured filter conditions using Filters DSL
    - top_k: Max results (default: -1). -1 = return all results meeting threshold (up to 100)
    - radius: Similarity threshold for vector/hybrid/rrf retrieval (0.0-1.0)
    - include_original_data: Whether to return original data (default: false)
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Search query text",
        examples=["What did Alice say about the project?"],
    )
    method: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_SEARCH_METHOD", "hybrid"),
        description="""Retrieval method:
- keyword: BM25 keyword retrieval (ES only)
- vector: Vector semantic retrieval (Milvus only)
- hybrid: Hybrid retrieval (default). episodic_memory uses hierarchical retrieval, others use ES + Milvus + Rerank
- agentic: LLM-guided multi-round retrieval
Default controlled by DEFAULT_SEARCH_METHOD env var.""",
        examples=["keyword", "vector", "hybrid", "agentic"],
    )
    memory_types: List[str] = Field(
        default_factory=lambda: [
            MemoryType.EPISODIC_MEMORY.value,
            MemoryType.PROFILE.value,
        ],
        description="""List of memory types to search:
- episodic_memory: Episodic memory (ES + Milvus)
- profile: User profile (Milvus only)
- raw_message: Raw unprocessed messages (ES only)
- agent_memory: Agent memory - cases and skills (ES + Milvus)""",
        examples=[[MemoryType.EPISODIC_MEMORY.value, MemoryType.PROFILE.value]],
    )
    top_k: int = Field(
        default=-1,
        description="Max results. -1 = return all meeting threshold (up to 100). Valid: -1 or 1-100",
        ge=-1,
        le=100,
        examples=[10, -1],
    )
    radius: Optional[float] = Field(
        default=None,
        description="COSINE similarity threshold (0.0-1.0) for vector methods",
        ge=0.0,
        le=1.0,
        examples=[0.6],
    )
    include_original_data: bool = Field(
        default=False, description="Whether to return original data", examples=[False]
    )
    filters: Dict[str, Any] = Field(
        description="""Filter conditions using Filters DSL.
Must contain at least one of user_id or group_id at first level.
Supported fields: user_id, group_id, session_id, timestamp.
Operators: eq (implicit), in, gt, gte, lt, lte.
Combinators: AND, OR.

Examples:
{"user_id": "user_123", "group_id": {"in": ["group_a", "group_b"]}}
{"AND": [{"timestamp": {"gte": 1704067200000}}, {"timestamp": {"lt": 1740614399000}}]}"""
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if not v:
            return os.getenv("DEFAULT_SEARCH_METHOD", "hybrid")
        allowed = {"keyword", "vector", "hybrid", "agentic"}
        if v not in allowed:
            raise ValueError(
                f"Unknown method: '{v}'. Allowed: {', '.join(sorted(allowed))}"
            )
        return v

    @field_validator("memory_types")
    @classmethod
    def validate_memory_types(cls, v: List[str]) -> List[str]:
        allowed = {
            MemoryType.EPISODIC_MEMORY.value,
            MemoryType.PROFILE.value,
            MemoryType.RAW_MESSAGE.value,
            MemoryType.AGENT_MEMORY.value,
        }
        invalid = [mt for mt in v if mt not in allowed]
        if invalid:
            raise ValueError(
                f"memory_types must be from: {', '.join(sorted(allowed))}. "
                f"Invalid: {invalid}"
            )
        return v

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Recursively check for user_id or group_id in filters
        def has_user_or_group_filter(filters: Dict[str, Any]) -> bool:
            if "user_id" in filters or "group_id" in filters:
                return True
            # Check nested AND/OR conditions
            for key in ["AND", "OR"]:
                if key in filters and isinstance(filters[key], list):
                    for item in filters[key]:
                        if isinstance(item, dict) and has_user_or_group_filter(item):
                            return True
            return False

        if not has_user_or_group_filter(v):
            raise ValueError(
                "filters must contain at least one of 'user_id' or 'group_id'"
            )
        return v

    @model_validator(mode="after")
    def validate_request(self) -> "SearchMemoriesRequest":
        if self.top_k == 0:
            raise ValueError(
                "top_k must be -1 (return all results) or a positive integer (1-100)"
            )
        if self.top_k > MAX_RETRIEVE_LIMIT:
            object.__setattr__(self, "top_k", MAX_RETRIEVE_LIMIT)
        return self


class SearchQueryInfo(BaseModel):
    """Query information echoed in response"""

    text: str = Field(description="Search query text")
    method: str = Field(description="Retrieval method used")
    filters_applied: Optional[Dict[str, Any]] = Field(
        default=None, description="Filters that were applied"
    )


class SearchAtomicFactItem(BaseModel):
    """Atomic fact item in search response (with score)

    Derived from v1_atomic_fact_records collection.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    atomic_fact: Optional[str] = Field(
        default=None, description="Atomic fact content (single sentence)"
    )
    timestamp: Optional[datetime] = Field(
        default=None, description="Event occurrence time"
    )
    participants: Optional[List[str]] = Field(
        default=None, description="Related participant IDs"
    )
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")
    score: Optional[float] = Field(
        default=None, description="Relevance score (BM25 score, unbounded)"
    )
    parent_episode_id: Optional[str] = Field(
        default=None, description="Source episode ID (MRAG expansion)"
    )
    original_text: Optional[str] = Field(
        default=None, description="Original text from parent episode"
    )


class SearchEpisodeItem(BaseModel):
    """Episode item in search response (with score)

    Derived from v1_episodic_memories collection.
    Same as EpisodeItem but with optional score field and optional text fields.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(
        default=None, description="Owner user ID, null = group memory"
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: Optional[datetime] = Field(
        default=None, description="Event occurrence time"
    )
    participants: Optional[List[str]] = Field(
        default=None, description="Event participant IDs"
    )
    summary: Optional[str] = Field(default=None, description="Memory summary")
    subject: Optional[str] = Field(default=None, description="Memory subject")
    episode: Optional[str] = Field(
        default=None, description="Full episodic memory text"
    )
    type: Optional[str] = Field(default=None, description="Episode type")
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")
    score: Optional[float] = Field(
        default=None, description="Relevance score (BM25 score, unbounded)"
    )
    atomic_facts: List[SearchAtomicFactItem] = Field(
        default_factory=list, description="Atomic facts expanded from this episode"
    )


class SearchProfileItem(BaseModel):
    """Profile item in search response (with score)

    Derived from v1_user_profiles collection.
    Same as ProfileItem but with optional score field.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    profile_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Profile data"
    )
    scenario: Optional[str] = Field(
        default=None, description="Scenario type: solo or team"
    )
    memcell_count: Optional[int] = Field(default=None, description="Number of MemCells")
    score: Optional[float] = Field(
        default=None, description="Relevance score (BM25 score, unbounded)"
    )


class SearchForesightItem(BaseModel):
    """Foresight item in search response (with score)

    Derived from v1_foresight_records collection.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    sender_id: Optional[str] = Field(default=None, description="Sender ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    content: Optional[str] = Field(default=None, description="Foresight content")
    evidence: Optional[str] = Field(
        default=None, description="Evidence supporting this foresight"
    )
    start_time: Optional[str] = Field(
        default=None, description="Foresight start time (date string)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Foresight end time (date string)"
    )
    duration_days: Optional[int] = Field(default=None, description="Duration in days")
    timestamp: Optional[datetime] = Field(
        default=None, description="Creation timestamp"
    )
    participants: Optional[List[str]] = Field(
        default=None, description="Related participant IDs"
    )
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")
    score: Optional[float] = Field(
        default=None, description="Relevance score (BM25 score, unbounded)"
    )


class AgentCaseItem(BaseModel):
    """Agent case item in GET response.

    Derived from v1_agent_cases collection.
    No score field — use SearchAgentCaseItem for search responses.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    task_intent: Optional[str] = Field(
        default=None, description="Rewritten task intent as retrieval key"
    )
    approach: Optional[str] = Field(
        default=None, description="Step-by-step approach with decisions and lessons"
    )
    quality_score: Optional[float] = Field(
        default=None, description="Task completion quality score (0.0-1.0)"
    )
    key_insight: Optional[str] = Field(
        default=None, description="Pivotal strategy shift or decision"
    )
    timestamp: Optional[datetime] = Field(
        default=None, description="Task occurrence time"
    )
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")


class AgentSkillItem(BaseModel):
    """Agent skill item in GET response.

    Derived from v1_agent_skills collection.
    No score field — use SearchAgentSkillItem for search responses.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    cluster_id: Optional[str] = Field(default=None, description="MemScene cluster ID")
    name: Optional[str] = Field(default=None, description="Skill name")
    description: Optional[str] = Field(
        default=None, description="What this skill does and when to use it"
    )
    content: Optional[str] = Field(default=None, description="Full skill content")
    confidence: Optional[float] = Field(
        default=None, description="Confidence score (0.0-1.0)"
    )
    maturity_score: Optional[float] = Field(
        default=None, description="Maturity score (0.0-1.0)"
    )
    source_case_ids: List[str] = Field(
        default_factory=list,
        description="AgentCase IDs that triggered add/update of this skill",
    )


class SearchAgentCaseItem(BaseModel):
    """Agent case item in search response (with score).

    Same as AgentCaseItem but with optional score field for search ranking.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    task_intent: Optional[str] = Field(
        default=None, description="Rewritten task intent as retrieval key"
    )
    approach: Optional[str] = Field(
        default=None, description="Step-by-step approach with decisions and lessons"
    )
    quality_score: Optional[float] = Field(
        default=None, description="Task completion quality score (0.0-1.0)"
    )
    key_insight: Optional[str] = Field(
        default=None, description="Pivotal strategy shift or decision"
    )
    timestamp: Optional[datetime] = Field(
        default=None, description="Task occurrence time"
    )
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")
    score: Optional[float] = Field(
        default=None, description="Relevance score (search only)"
    )


class SearchAgentSkillItem(BaseModel):
    """Agent skill item in search response (with score).

    Same as AgentSkillItem but with optional score field for search ranking.
    """

    id: str = Field(description="MongoDB ObjectId as string")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    cluster_id: Optional[str] = Field(default=None, description="MemScene cluster ID")
    name: Optional[str] = Field(default=None, description="Skill name")
    description: Optional[str] = Field(
        default=None, description="What this skill does and when to use it"
    )
    content: Optional[str] = Field(default=None, description="Full skill content")
    confidence: Optional[float] = Field(
        default=None, description="Confidence score (0.0-1.0)"
    )
    maturity_score: Optional[float] = Field(
        default=None, description="Maturity score (0.0-1.0)"
    )
    source_case_ids: List[str] = Field(
        default_factory=list,
        description="AgentCase IDs that triggered add/update of this skill",
    )
    score: Optional[float] = Field(
        default=None, description="Relevance score (search only)"
    )


class AgentMemorySearchResult(BaseModel):
    """Agent memory search result container.

    Groups agent cases and skills under one structure
    when memory_types includes 'agent_memory'.
    """

    cases: List[SearchAgentCaseItem] = Field(
        default_factory=list, description="Agent case search results"
    )
    skills: List[SearchAgentSkillItem] = Field(
        default_factory=list, description="Agent skill search results"
    )


class SearchMemoriesResponseData(BaseModel):
    """Memory search response data (v1)

    Result data for POST /api/v1/memories/search endpoint.
    Wrapped in envelope: { "data": SearchMemoriesResponseData }
    """

    episodes: List[SearchEpisodeItem] = Field(
        default_factory=list, description="Episodic memory search results"
    )
    profiles: List[SearchProfileItem] = Field(
        default_factory=list, description="Profile search results"
    )
    raw_messages: List[RawMessageDTO] = Field(
        default_factory=list, description="Raw unprocessed messages (pending)"
    )
    agent_memory: Optional[AgentMemorySearchResult] = Field(
        default=None,
        description="Agent memory search results containing cases and skills "
        "(populated when memory_types includes 'agent_memory')",
    )
    query: SearchQueryInfo = Field(description="Query information echoed from request")
    original_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Original data (if include_original_data=true)"
    )


class SearchMemoriesResponse(BaseApiResponse[SearchMemoriesResponseData]):
    """Memory search response (v1)

    Response for POST /api/v1/memories/search endpoint.
    """

    data: SearchMemoriesResponseData = Field(description="Memory search result")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": {
                        "episodes": [
                            {
                                "id": "67c8a1b2f3e4d5c6a7b8c9d0",
                                "user_id": "user_123",
                                "group_id": "group_abc",
                                "session_id": "sess_abc",
                                "timestamp": "2026-03-01T10:00:00Z",
                                "participants": ["user_123", "user_456"],
                                "summary": "Team discussed Q1 roadmap",
                                "subject": "Q1 Roadmap Discussion",
                                "episode": "Alice and Bob discussed the Q1 roadmap...",
                                "type": "Conversation",
                                "parent_type": "memcell",
                                "parent_id": "67c8a1b2f3e4d5c6a7b8c9d1",
                                "score": 0.85,
                            }
                        ],
                        "profiles": [],
                        "query": {
                            "text": "What did Alice say about the project?",
                            "method": "hybrid",
                            "filters_applied": {
                                "user_id": "user_123",
                                "group_id": "group_abc",
                            },
                        },
                        "original_data": None,
                    }
                },
                {
                    "data": {
                        "episodes": [],
                        "profiles": [
                            {
                                "id": "67c8a1b2f3e4d5c6a7b8c9e0",
                                "user_id": "user_456",
                                "group_id": "group_abc",
                                "profile_data": {
                                    "explicit_info": {"Role": "Product Manager"},
                                    "implicit_traits": {"Leadership": "Proactive"},
                                },
                                "scenario": "team",
                                "memcell_count": 150,
                                "score": 0.72,
                            }
                        ],
                        "query": {
                            "text": "What is Alice's role?",
                            "method": "vector",
                            "filters_applied": {"user_id": "user_456"},
                        },
                        "original_data": None,
                    }
                },
                {
                    "data": {
                        "episodes": [],
                        "profiles": [],
                        "agent_memory": {
                            "cases": [
                                {
                                    "id": "67d1a2b3c4e5f6a7b8c9d0e1",
                                    "user_id": "user_01",
                                    "session_id": "sess_agent_001",
                                    "task_intent": "Handle API timeout errors with retry and fallback",
                                    "approach": "1. Catch timeout exception. 2. Retry up to 3 times with exponential backoff. 3. If all retries fail, return cached result or error message.",
                                    "quality_score": 0.95,
                                    "timestamp": "2026-03-15T14:30:00Z",
                                    "score": 0.88,
                                }
                            ],
                            "skills": [
                                {
                                    "id": "67d2b3c4d5e6f7a8b9c0d1e2",
                                    "user_id": "user_01",
                                    "cluster_id": "cluster_error_handling",
                                    "name": "API Error Handling with Retry",
                                    "description": "Handle API errors with exponential backoff retry and graceful fallback",
                                    "content": "Pattern: try/except with max_retries=3, backoff_factor=2. On final failure, return cached data or user-friendly error.",
                                    "confidence": 0.91,
                                    "maturity_score": 0.82,
                                    "score": 0.79,
                                }
                            ],
                        },
                        "query": {
                            "text": "How to handle timeout errors",
                            "method": "hybrid",
                            "filters_applied": {"user_id": "user_01"},
                        },
                        "original_data": None,
                    }
                },
            ]
        }
    }
