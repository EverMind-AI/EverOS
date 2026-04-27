"""
Session Beanie ODM model

Session container management for V1 API.
"""

from datetime import datetime
from typing import Optional
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBase,
)
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING


class Session(TenantAwareDocumentBase, AuditBase):
    """
    Session document model

    Stores session information for conversation isolation.
    Auto-registered when Add Memory receives a session_id.
    """

    session_id: str = Field(..., description="Session identifier (unique)")
    name: Optional[str] = Field(default=None, description="Session display name")
    description: Optional[str] = Field(default=None, description="Session description")

    model_config = ConfigDict(
        collection="v1_sessions",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "session_id": "sess_abc",
                "name": "Morning Chat",
                "description": "Casual morning conversation",
            }
        },
        extra="allow",
    )

    class Settings:
        """Beanie settings"""

        name = "v1_sessions"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("session_id", ASCENDING)],
                name="idx_tenant_session_id",
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
