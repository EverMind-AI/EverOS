"""
ForesightRecord Beanie ODM model

Unified storage of foresights extracted from episodic memories (personal or group).
"""

from datetime import datetime
from typing import List, Optional
from core.oxm.mongo.document_base import DocumentBase
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBaseWithSoftDelete,
)
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId
from api_specs.memory_types import ParentType


class ForesightRecord(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    Generic foresight document model

    Unified storage of foresight information extracted from personal or group episodic memories.
    When user_id exists, it represents personal foresight; when user_id is empty and group_id exists, it represents group foresight.
    """

    # field from api input
    user_id: Optional[str] = Field(
        default=None,
        description="User ID, required for personal memory, None for group memory",
    )
    # field from api input
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    content: str = Field(..., min_length=1, description="Foresight content")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Evidence
    evidence: Optional[str] = Field(
        default=None, description="Evidence supporting this foresight"
    )

    # Time range fields
    start_time: Optional[str] = Field(
        default=None, description="Foresight start time (date string, e.g., 2024-01-01)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Foresight end time (date string, e.g., 2024-12-31)"
    )
    duration_days: Optional[int] = Field(default=None, description="Duration in days")

    type: Optional[str] = Field(
        default=None, description="Foresight type, such as Conversation"
    )

    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of messages"
    )

    # Vector and model
    vector: Optional[List[float]] = Field(
        default=None, description="Text vector of the foresight"
    )
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    model_config = ConfigDict(
        collection="v1_foresight_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "id": "foresight_001",
                "user_id": "user_12345",
                "content": "User likes Sichuan cuisine, especially spicy hotpot",
                "parent_type": ParentType.MEMCELL.value,
                "parent_id": "memcell_001",
                "start_time": "2024-01-01",
                "end_time": "2024-12-31",
                "duration_days": 365,
                "group_id": "group_friends",
                "participants": ["Zhang San", "Li Si"],
                "vector": [0.1, 0.2, 0.3],
                "vector_model": "text-embedding-3-small",
                "evidence": "Mentioned multiple times in chat about liking hotpot",
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id

    class Settings:
        """Beanie settings"""

        name = "v1_foresight_records"

        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("deleted_at", ASCENDING)],
                name="idx_tenant_deleted_at",
                sparse=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("parent_id", ASCENDING)],
                name="idx_tenant_parent_id",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("start_time", ASCENDING),
                    ("end_time", ASCENDING),
                ],
                name="idx_tenant_user_time_range",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("start_time", ASCENDING),
                    ("end_time", ASCENDING),
                ],
                name="idx_tenant_group_time_range",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("start_time", ASCENDING),
                    ("end_time", ASCENDING),
                ],
                name="idx_tenant_group_user_time_range",
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


class ForesightRecordProjection(DocumentBase, AuditBase):
    """
    Simplified foresight model (without vector)

    Used in most scenarios where vector data is not needed, reducing data transfer and memory usage.
    """

    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    user_id: Optional[str] = Field(
        default=None,
        description="User ID, required for personal memory, None for group memory",
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    sender_id: Optional[str] = Field(default=None, description="Sender identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    content: str = Field(..., min_length=1, description="Foresight content")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time range fields
    start_time: Optional[str] = Field(
        default=None, description="Foresight start time (date string, e.g., 2024-01-01)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Foresight end time (date string, e.g., 2024-12-31)"
    )
    duration_days: Optional[int] = Field(default=None, description="Duration in days")

    type: Optional[str] = Field(
        default=None, description="Foresight type, such as Conversation"
    )

    # Participant information
    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of related participants"
    )

    # Vector model information (retain model name, but exclude vector data)
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    # Evidence
    evidence: Optional[str] = Field(
        default=None, description="Evidence supporting this foresight"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat(), PydanticObjectId: str},
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id


# Export models
__all__ = ["ForesightRecord", "ForesightRecordProjection"]
