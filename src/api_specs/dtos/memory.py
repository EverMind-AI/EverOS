"""Memory resource DTOs.

This module contains DTOs related to memory CRUD operations:
- Memorize (POST /api/v1/memories)
- Fetch (GET /api/v1/memories)
- Search (GET /api/v1/memories/search)
- Delete (DELETE /api/v1/memories)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import re

from bson import ObjectId
from pydantic import BaseModel, Field, model_validator, SkipValidation, SerializeAsAny

from api_specs.dtos.base import BaseApiResponse
from api_specs.memory_types import BaseMemory, RawDataType
from api_specs.memory_models import (
    MemoryType,
    Metadata,
    MemoryModel,
    RetrieveMethod,
    MessageSenderRole,
)
from core.oxm.constants import MAGIC_ALL, MAX_FETCH_LIMIT, MAX_RETRIEVE_LIMIT


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
# Memorize DTOs (POST /api/v1/memories)
# =============================================================================


class MemorizeRequest(BaseModel):
    """Memory storage request (internal business layer)"""

    history_raw_data_list: list[RawData]
    new_raw_data_list: list[RawData]
    raw_data_type: RawDataType
    # Full list of user_id for the entire group
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    current_time: Optional[datetime] = None
    # Optional extraction control parameters
    enable_foresight_extraction: bool = True  # Whether to extract foresight
    enable_event_log_extraction: bool = True  # Whether to extract event logs

    model_config = {"arbitrary_types_allowed": True}


class MemorizeMessageRequest(BaseModel):
    """
    Store single message request body (HTTP API layer)

    Used for POST /api/v1/memories endpoint
    """

    group_id: Optional[str] = Field(
        default=None,
        description="Group ID. If not provided, will automatically generate based on hash(sender) + '_group' suffix, "
        "representing single-user mode where each user's messages are extracted into separate memory spaces.",
        examples=["group_123"],
    )
    group_name: Optional[str] = Field(
        default=None, description="Group name", examples=["Project Discussion Group"]
    )
    message_id: str = Field(
        ..., description="Message unique identifier", examples=["msg_001"]
    )
    create_time: str = Field(
        ...,
        description="Message creation time (ISO 8601 format)",
        examples=["2025-01-15T10:00:00+00:00"],
    )
    sender: str = Field(
        ...,
        description="Sender user ID (required). Also used as user_id internally for memory ownership.",
        examples=["user_001"],
    )
    sender_name: Optional[str] = Field(
        default=None,
        description="Sender name (uses sender if not provided)",
        examples=["John"],
    )
    role: Optional[str] = Field(
        default=None,
        description="""Message sender role, used to identify the source of the message.
Enum values from MessageSenderRole:
- user: Message from a human user
- assistant: Message from an AI assistant""",
        examples=["user", "assistant"],
    )
    content: str = Field(
        ...,
        description="Message content",
        examples=["Let's discuss the technical solution for the new feature today"],
    )
    refer_list: Optional[List[str]] = Field(
        default=None,
        description="List of referenced message IDs",
        examples=[["msg_000"]],
    )

    @model_validator(mode="after")
    def validate_role(self):
        """Validate that role is a valid MessageSenderRole value"""
        if self.role is not None and not MessageSenderRole.is_valid(self.role):
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {[r.value for r in MessageSenderRole]}"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "With explicit group_id (multi-user group mode)",
                    "value": {
                        "group_id": "group_123",
                        "group_name": "Project Discussion Group",
                        "message_id": "msg_001",
                        "create_time": "2025-01-15T10:00:00+00:00",
                        "sender": "user_001",
                        "sender_name": "John",
                        "role": "user",
                        "content": "Let's discuss the technical solution for the new feature today",
                        "refer_list": ["msg_000"],
                    },
                },
                {
                    "summary": "Without group_id (single-user mode, auto-generated)",
                    "value": {
                        "message_id": "msg_002",
                        "create_time": "2025-01-15T10:05:00+00:00",
                        "sender": "user_001",
                        "sender_name": "John",
                        "role": "user",
                        "content": "What's the weather like today?",
                    },
                },
            ]
        }
    }


class MemorizeResult(BaseModel):
    """Memory storage result data

    Result data for POST /api/v1/memories endpoint.
    """

    saved_memories: List[Any] = Field(
        default_factory=list,
        description="List of saved memories (fetch via API for details)",
    )
    count: int = Field(
        default=0, description="Number of memories extracted", examples=[1, 0]
    )
    status_info: str = Field(
        default="accumulated",
        description="Processing status: 'extracted' (memories created) or 'accumulated' (waiting for boundary)",
        examples=["extracted", "accumulated"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Extracted memories (boundary triggered)",
                    "value": {
                        "saved_memories": [],
                        "count": 1,
                        "status_info": "extracted",
                    },
                },
                {
                    "summary": "Message queued (boundary not triggered)",
                    "value": {
                        "saved_memories": [],
                        "count": 0,
                        "status_info": "accumulated",
                    },
                },
            ]
        }
    }


class MemorizeResponse(BaseApiResponse[MemorizeResult]):
    """Memory storage response

    Response for POST /api/v1/memories endpoint.
    """

    result: MemorizeResult = Field(
        default_factory=MemorizeResult, description="Memory storage result"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Extracted memories (boundary triggered)",
                    "value": {
                        "status": "ok",
                        "message": "Extracted 1 memories",
                        "result": {
                            "saved_memories": [],
                            "count": 1,
                            "status_info": "extracted",
                        },
                    },
                },
                {
                    "summary": "Message queued (boundary not triggered)",
                    "value": {
                        "status": "ok",
                        "message": "Message queued, awaiting boundary detection",
                        "result": {
                            "saved_memories": [],
                            "count": 0,
                            "status_info": "accumulated",
                        },
                    },
                },
            ]
        }
    }


# =============================================================================
# Fetch DTOs (GET /api/v1/memories)
# =============================================================================


class FetchMemRequest(BaseModel):
    """Memory fetch request

    Used for GET /api/v1/memories endpoint.

    Note:
    - user_id and group_id support special value MAGIC_ALL ("__all__") to skip filtering
    - Empty string or None for user_id/group_id filters for null/empty values
    - user_id and group_id cannot both be MAGIC_ALL
    - limit is capped at MAX_FETCH_LIMIT (500)
    """

    user_id: Optional[str] = Field(
        default=None, description="User ID", examples=["user_123"]
    )
    group_id: Optional[str] = Field(
        default=None, description="Group ID", examples=["group_456"]
    )
    limit: Optional[int] = Field(
        default=40,
        description="Maximum number of memories to return",
        ge=1,
        le=500,
        examples=[20],
    )
    offset: Optional[int] = Field(
        default=0, description="Pagination offset", ge=0, examples=[0]
    )
    memory_type: Optional[MemoryType] = Field(
        default=MemoryType.EPISODIC_MEMORY,
        description="""Memory type, enum values from MemoryType:
- profile: user profile
- episodic_memory: episodic memory (default)
- foresight: prospective memory
- event_log: event log (atomic facts)""",
        examples=["episodic_memory"],
    )
    version_range: Optional[Tuple[Optional[str], Optional[str]]] = Field(
        default=None,
        description="Version range filter, format (start, end), closed interval",
        examples=[("v1.0", "v2.0")],
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Start time for time range filtering (ISO 8601 format)",
        examples=["2024-01-01T00:00:00"],
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time for time range filtering (ISO 8601 format)",
        examples=["2024-12-31T23:59:59"],
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_request(self) -> "FetchMemRequest":
        """Validate request parameters"""
        if self.user_id == MAGIC_ALL and self.group_id == MAGIC_ALL:
            raise ValueError("user_id and group_id cannot both be MAGIC_ALL")

        # Cap limit at MAX_FETCH_LIMIT
        if self.limit and self.limit > MAX_FETCH_LIMIT:
            object.__setattr__(self, "limit", MAX_FETCH_LIMIT)

        return self

    def get_memory_types(self) -> List[MemoryType]:
        """Get the list of memory types to query"""
        return [self.memory_type]


class FetchMemResponse(BaseModel):
    """Memory fetch response (result data)"""

    memories: SkipValidation[List[MemoryModel]] = Field(default_factory=list)
    total_count: int = 0
    has_more: bool = False
    metadata: SkipValidation[Optional[Metadata]] = None

    model_config = {"arbitrary_types_allowed": True}


class FetchMemoriesResponse(BaseApiResponse[FetchMemResponse]):
    """Memory fetch API response

    Response for GET /api/v1/memories endpoint.
    """

    result: FetchMemResponse = Field(description="Memory fetch result")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "Memory retrieval successful, retrieved 1 memories",
                "result": {
                    "memories": [
                        {
                            "memory_type": "episodic_memory",
                            "user_id": "user_123",
                            "timestamp": "2024-01-15T10:30:00",
                            "content": "User discussed coffee during the project sync",
                            "summary": "Project sync coffee note",
                        }
                    ],
                    "total_count": 100,
                    "has_more": False,
                    "metadata": {
                        "source": "fetch_mem_service",
                        "user_id": "user_123",
                        "memory_type": "fetch",
                    },
                },
            }
        }
    }


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
    group_id: Optional[str] = Field(
        default=None,
        description="Group ID (at least one of user_id or group_id must be provided)",
        examples=["group_456"],
    )
    memory_types: List[MemoryType] = Field(
        default_factory=list,
        description="""List of memory types to retrieve, enum values from MemoryType:
- episodic_memory: episodic memory
- foresight: prospective memory
- event_log: event log (atomic facts)
Note: profile type is not supported in search interface""",
        examples=[["episodic_memory"]],
    )
    top_k: int = Field(
        default=40,
        description="Maximum number of results to return",
        ge=1,
        le=100,
        examples=[10],
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata", examples=[True]
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Time range start (ISO 8601 format)",
        examples=["2024-01-01T00:00:00"],
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Time range end (ISO 8601 format)",
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
    current_time: Optional[str] = Field(
        default=None,
        description="Current time, used to filter forward-looking events within validity period",
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
        if self.user_id == MAGIC_ALL and self.group_id == MAGIC_ALL:
            raise ValueError("user_id and group_id cannot both be MAGIC_ALL")

        if self.top_k and self.top_k > MAX_RETRIEVE_LIMIT:
            object.__setattr__(self, "top_k", MAX_RETRIEVE_LIMIT)

        return self


class PendingMessage(BaseModel):
    """Pending message that has not yet been extracted into memory.

    Represents a cached message waiting for boundary detection or memory extraction.
    """

    id: str  # MongoDB ObjectId as string
    request_id: str  # Request ID
    message_id: Optional[str] = None  # Message ID
    group_id: Optional[str] = None  # Group ID
    user_id: Optional[str] = None  # User ID
    sender: Optional[str] = None  # Sender ID
    sender_name: Optional[str] = None  # Sender name
    group_name: Optional[str] = None  # Group name
    content: Optional[str] = None  # Message content
    refer_list: Optional[List[str]] = None  # List of referenced message IDs
    message_create_time: Optional[str] = None  # Message creation time (ISO 8601 format)
    created_at: Optional[str] = None  # Record creation time (ISO 8601 format)
    updated_at: Optional[str] = None  # Record update time (ISO 8601 format)


class RetrieveMemResponse(BaseModel):
    """Memory retrieve/search response (result data)"""

    memories: SerializeAsAny[SkipValidation[List[Dict[str, List[BaseMemory]]]]] = Field(
        default_factory=list
    )
    scores: SkipValidation[List[Dict[str, List[float]]]] = Field(default_factory=list)
    importance_scores: List[float] = Field(default_factory=list)
    original_data: SkipValidation[List[Dict[str, List[Dict[str, Any]]]]] = Field(
        default_factory=list
    )
    total_count: int = 0
    has_more: bool = False
    query_metadata: SkipValidation[Optional[Metadata]] = None
    metadata: SkipValidation[Optional[Metadata]] = None
    pending_messages: SkipValidation[List[PendingMessage]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class SearchMemoriesResponse(BaseApiResponse[RetrieveMemResponse]):
    """Memory search API response

    Response for GET /api/v1/memories/search endpoint.
    """

    result: RetrieveMemResponse = Field(description="Memory search result")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "Memory search successful, retrieved 1 groups",
                "result": {
                    "memories": [
                        {
                            "episodic_memory": [
                                {
                                    "memory_type": "episodic_memory",
                                    "user_id": "user_123",
                                    "timestamp": "2024-01-15T10:30:00",
                                    "summary": "Discussed coffee choices",
                                    "group_id": "group_456",
                                }
                            ]
                        }
                    ],
                    "scores": [{"episodic_memory": [0.95]}],
                    "importance_scores": [0.85],
                    "original_data": [],
                    "total_count": 45,
                    "has_more": False,
                    "query_metadata": {
                        "source": "episodic_memory_es_repository",
                        "user_id": "user_123",
                        "memory_type": "retrieve",
                    },
                    "metadata": {
                        "source": "episodic_memory_es_repository",
                        "user_id": "user_123",
                        "memory_type": "retrieve",
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
    - event_id, user_id, group_id are combined filter conditions
    - If all three are provided, all conditions must be met
    - If not provided, use MAGIC_ALL ("__all__") to skip filtering
    - Cannot all be MAGIC_ALL (at least one filter required)
    """

    event_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="Memory event_id (filter condition)",
        examples=["507f1f77bcf86cd799439011", MAGIC_ALL],
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
        # Check if all are MAGIC_ALL
        if (
            self.event_id == MAGIC_ALL
            and self.user_id == MAGIC_ALL
            and self.group_id == MAGIC_ALL
        ):
            raise ValueError(
                "At least one of event_id, user_id, or group_id must be provided (not MAGIC_ALL)"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Delete by event_id only",
                    "value": {
                        "event_id": "507f1f77bcf86cd799439011",
                        "user_id": MAGIC_ALL,
                        "group_id": MAGIC_ALL,
                    },
                },
                {
                    "summary": "Delete by user_id only",
                    "value": {
                        "event_id": MAGIC_ALL,
                        "user_id": "user_123",
                        "group_id": MAGIC_ALL,
                    },
                },
                {
                    "summary": "Delete by user_id and group_id",
                    "value": {
                        "event_id": MAGIC_ALL,
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

    result: DeleteMemoriesResult = Field(description="Delete operation result")

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


class RestoreMemoriesResult(BaseModel):
    """Restore soft-deleted memories result data"""

    filters: List[str] = Field(
        default_factory=list,
        description="List of filter types used for restoration",
        examples=[["event_id"], ["user_id"]],
    )
    count: int = Field(
        default=0, description="Number of memories restored", examples=[1, 25]
    )


class RestoreMemoriesResponse(BaseApiResponse[RestoreMemoriesResult]):
    """Restore soft-deleted memories API response

    Response for POST /api/v1/memories/restore endpoint.
    """

    result: RestoreMemoriesResult = Field(description="Restore operation result")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Restore by event_id",
                    "value": {
                        "status": "ok",
                        "message": "Successfully restored 1 memory",
                        "result": {"filters": ["event_id"], "count": 1},
                    },
                },
                {
                    "summary": "Restore by user_id",
                    "value": {
                        "status": "ok",
                        "message": "Successfully restored 25 memories",
                        "result": {"filters": ["user_id"], "count": 25},
                    },
                },
            ]
        }
    }
