# -*- coding: utf-8 -*-
"""
RawMessage MongoDB Document Model

Stores individual messages from add/flush API requests.
Each message is stored as a separate document for incremental processing.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING

from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBaseWithSoftDelete,
)


class RawMessage(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    Raw Message Document Model

    Stores individual messages from API requests:
    - group_id: conversation group ID
    - request_id: request ID
    - message core fields: message_id, sender_id, sender_name, content, role, etc.
    """

    # Core fields
    group_id: str = Field(..., description="Conversation group ID")
    request_id: str = Field(..., description="Request ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

    # ========== Message core fields ==========
    message_id: Optional[str] = Field(default=None, description="Message ID")
    timestamp: Optional[str] = Field(
        default=None,
        description="Message timestamp (ISO 8601 format with timezone, e.g. 2025-01-15T10:00:00+00:00)",
    )
    sender_id: Optional[str] = Field(default=None, description="Sender ID")
    sender_name: Optional[str] = Field(default=None, description="Sender name")
    role: Optional[str] = Field(
        default=None,
        description="Message sender role: 'user' for human, 'assistant' / 'agent' for AI",
    )
    content_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Message content items list, e.g. [{type: 'text', content: '...'}, {type: 'image', ...}]",
    )

    # Agent-specific fields (OpenAI chat completion format)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made by assistant (OpenAI format), only when role='assistant'",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call ID this message responds to, only when role='tool'",
    )

    # Request metadata
    version: Optional[str] = Field(default=None, description="Code version")
    endpoint_name: Optional[str] = Field(default=None, description="Endpoint name")
    method: Optional[str] = Field(default=None, description="HTTP method")
    url: Optional[str] = Field(default=None, description="Request URL")

    # Original event ID (used to associate with RequestHistory)
    event_id: Optional[str] = Field(default=None, description="Original event ID")

    # Sync status field (numeric)
    # -1: log record only (raw message just saved)
    #  0: accumulating in window (confirmed entering accumulation window)
    #  1: already fully used (after boundary detection)
    sync_status: int = Field(
        default=-1,
        description="Sync status: -1=log record, 0=window accumulating, 1=already used",
    )

    model_config = ConfigDict(
        collection="v1_raw_messages",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "group_id": "group_123",
                "request_id": "req_456",
                "session_id": "-1",
                "message_id": "msg_001",
                "timestamp": "2024-01-01T12:00:00+00:00",
                "sender_id": "user_789",
                "sender_name": "Test User",
                "content_items": [
                    {"type": "text", "content": "This is a test message"}
                ],
                "version": "1.0.0",
                "endpoint_name": "add_personal_memories",
            }
        },
    )

    class Settings:
        """Beanie settings"""

        name = "v1_raw_messages"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("created_at", DESCENDING)],
                name="idx_tenant_created_at",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("event_id", ASCENDING)],
                name="idx_tenant_event_id",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name="idx_tenant_group_created",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("request_id", ASCENDING)],
                name="idx_tenant_request_id",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_group_timestamp",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("sync_status", ASCENDING),
                ],
                name="idx_tenant_group_sync_status",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("session_id", ASCENDING),
                    ("sync_status", ASCENDING),
                ],
                name="idx_tenant_group_session_sync",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("sender_id", ASCENDING),
                    ("sync_status", ASCENDING),
                ],
                name="idx_tenant_group_sender_sync",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("sender_id", ASCENDING),
                    ("message_id", ASCENDING),
                ],
                name="idx_tenant_group_sender_message",
            ),
        ]
