from datetime import datetime
from typing import Optional
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBase,
)
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase


class ConversationStatus(TenantAwareDocumentBase, AuditBase):
    """
    Conversation status document model

    Stores conversation status information, including group ID, message read time, etc.
    """

    # Basic information
    group_id: str = Field(..., description="Group ID, empty means private chat")
    session_id: Optional[str] = Field(default=None, description="Session ID")

    old_msg_start_time: Optional[datetime] = Field(
        default=None, description="Conversation window read start time"
    )
    new_msg_start_time: Optional[datetime] = Field(
        default=None, description="Accumulated new conversation read start time"
    )
    last_memcell_time: Optional[datetime] = Field(
        default=None, description="Accumulated memCell read start time"
    )

    model_config = ConfigDict(
        collection="v1_conversation_status",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "group_id": "group_001",
                "old_msg_start_time": datetime(2021, 1, 1, 0, 0, 0),
                "new_msg_start_time": datetime(2021, 1, 1, 0, 0, 0),
                "last_memcell_time": datetime(2021, 1, 1, 0, 0, 0),
            }
        },
        extra="allow",
    )

    @property
    def conversation_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "v1_conversation_status"
        indexes = [
            # Composite unique index: one status per (group_id, session_id) pair
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("session_id", ASCENDING),
                ],
                name="idx_tenant_group_session",
                unique=True,
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
        validate_on_save = True
        use_state_management = True
