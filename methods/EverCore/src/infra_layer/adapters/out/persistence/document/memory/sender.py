"""
Sender Beanie ODM model

User/agent identity management. Auto-created from Add Memory messages.
"""

from datetime import datetime
from typing import Optional
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBase,
)
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING


class Sender(TenantAwareDocumentBase, AuditBase):
    """
    Sender document model

    Stores user/agent identity information.
    Auto-registered when Add Memory receives a new sender_id.
    """

    sender_id: str = Field(..., description="Sender identifier (unique)")
    name: Optional[str] = Field(default=None, description="Sender display name")

    model_config = ConfigDict(
        collection="v1_senders",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={"example": {"sender_id": "user_123", "name": "Alice"}},
        extra="allow",
    )

    class Settings:
        """Beanie settings"""

        name = "v1_senders"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("sender_id", ASCENDING)],
                name="idx_tenant_sender_id",
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
