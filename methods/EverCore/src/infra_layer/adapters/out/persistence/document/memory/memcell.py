"""
MemCell Beanie ODM model

MemCell data model definition based on Beanie ODM, supporting MongoDB sharded clusters.
"""

from datetime import datetime
from typing import List, Optional
from enum import Enum

from beanie import Indexed
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBaseWithSoftDelete,
)
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase


class DataTypeEnum(str, Enum):
    """Data type enumeration"""

    CONVERSATION = "Conversation"
    AGENTCONVERSATION = "AgentConversation"


class MemCell(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    MemCell document model

    Storage model for scene segmentation results, supporting flexible extension and high-performance queries.

    Supports soft delete functionality:
    - Use delete() method for soft deletion
    - Use find_one(), find_many() to automatically filter out deleted records
    - Use hard_find_one(), hard_find_many() to query including deleted records
    - Use hard_delete() for physical deletion
    """

    # Core fields
    timestamp: Indexed(datetime) = Field(..., description="Occurrence time, shard key")

    # Optional fields
    group_id: Optional[Indexed(str)] = Field(
        default=None, description="Group ID, empty means private chat"
    )
    session_id: Optional[str] = Field(default=None, description="Session ID")
    original_data: Optional[List] = Field(
        default=None, description="Original information"
    )
    # NOTE: participants and sender_ids currently hold the same values (both are sender_id).
    # participants is not yet implemented as display names; it is populated with sender_ids
    # as a placeholder. Once display-name resolution is available, participants will carry
    # human-readable names while sender_ids will remain the raw identifiers.
    participants: Optional[List[str]] = Field(
        default=None, description="Names of event participants"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of event participants"
    )
    type: Optional[DataTypeEnum] = Field(default=None, description="Scenario type")

    model_config = ConfigDict(
        # Collection name
        collection="v1_memcells",
        # Validation configuration
        validate_assignment=True,
        # JSON serialization configuration
        json_encoders={datetime: lambda dt: dt.isoformat()},
        # Example data
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "group_67890",
                "timestamp": "2024-12-01T10:30:00.000Z",
                "original_data": [
                    {
                        "message": {
                            "message_id": "msg_001",
                            "sender_id": "user_123",
                            "sender_name": "Alice",
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "content": "Let's discuss the new feature design",
                                }
                            ],
                            "timestamp": "2025-01-15T10:00:00+00:00",
                        }
                    }
                ],
                "participants": ["user_123", "user_456"],
                "sender_ids": ["user_123", "user_456"],
                "type": "Conversation",
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        # Collection name
        name = "v1_memcells"

        # Index definitions
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("deleted_at", ASCENDING)],
                name="idx_tenant_deleted_at",
                sparse=True,
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_user_deleted_timestamp",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_group_deleted_timestamp",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("participants", ASCENDING)],
                name="idx_tenant_participants",
                sparse=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("sender_ids", ASCENDING)],
                name="idx_tenant_sender_ids",
                sparse=True,
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("type", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_user_type_deleted_timestamp",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("type", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_group_type_deleted_timestamp",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("created_at", DESCENDING)],
                name="idx_tenant_created_at",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("updated_at", DESCENDING)],
                name="idx_tenant_updated_at",
            ),
        ]

        # Validation settings
        validate_on_save = True
        use_state_management = True


# Export models
__all__ = ["MemCell", "DataTypeEnum"]
