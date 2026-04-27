"""
Group Beanie ODM model

Group container management for V1 API.
"""

from datetime import datetime
from typing import Optional
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBase,
)
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING


class Group(TenantAwareDocumentBase, AuditBase):
    """
    Group document model

    Stores group information.
    Auto-registered when Add Memory receives a group_id.
    """

    group_id: str = Field(..., description="Group identifier (unique)")
    name: Optional[str] = Field(default=None, description="Group display name")
    description: Optional[str] = Field(default=None, description="Group description")

    model_config = ConfigDict(
        collection="v1_groups",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "group_id": "group_abc",
                "name": "Project Discussion",
                "description": "Weekly sync on Project X",
            }
        },
        extra="allow",
    )

    class Settings:
        """Beanie settings"""

        name = "v1_groups"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("group_id", ASCENDING)],
                name="idx_tenant_group_id",
                unique=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("name", ASCENDING)], name="idx_tenant_name"
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
