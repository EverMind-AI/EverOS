"""
AtomicFactRecord Beanie ODM model

Unified storage for atomic facts (atomic facts) extracted from episodic memory (individual or group).
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.mongo.document_base import DocumentBase
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBaseWithSoftDelete,
)
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId
from api_specs.memory_types import ParentType


class AtomicFactRecord(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    Generic atomic fact document model

    Stores atomic facts split from individual or group episodic memory for fine-grained retrieval.
    """

    # field from api input
    user_id: Optional[str] = Field(
        default=None, description="User ID, required for personal events"
    )
    # field from api input
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    atomic_fact: str = Field(..., description="Atomic fact content (single sentence)")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time information
    timestamp: datetime = Field(..., description="Event occurrence time")

    type: Optional[str] = Field(
        default=None, description="Event type, such as Conversation"
    )

    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of messages"
    )

    # Vector and model
    vector: Optional[List[float]] = Field(
        default=None, description="Atomic fact vector"
    )
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    model_config = ConfigDict(
        collection="v1_atomic_fact_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "id": "atomic_fact_001",
                "user_id": "user_12345",
                "atomic_fact": "The user went to Chengdu on January 1, 2024, and enjoyed the local Sichuan cuisine.",
                "parent_type": ParentType.MEMCELL.value,
                "parent_id": "memcell_001",
                "timestamp": "2024-01-01T10:00:00+00:00",
                "group_id": "group_travel",
                "participants": ["Zhang San", "Li Si"],
                "vector": [0.1, 0.2, 0.3],
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id

    class Settings:
        """Beanie Settings"""

        name = "v1_atomic_fact_records"

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
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_user_timestamp",
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
                    ("user_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_tenant_group_user_timestamp",
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


class AtomicFactRecordProjection(DocumentBase, AuditBase):
    """
    Simplified atomic fact model (without vector)

    Used in most scenarios where vector data is not needed, reducing data transfer and memory usage.
    """

    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    user_id: Optional[str] = Field(
        default=None, description="User ID, required for personal events"
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    sender_id: Optional[str] = Field(default=None, description="Sender identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    atomic_fact: str = Field(..., description="Atomic fact content (single sentence)")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time information
    timestamp: datetime = Field(..., description="Event occurrence time")

    type: Optional[str] = Field(
        default=None, description="Event type, such as Conversation"
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

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
            PydanticObjectId: lambda oid: str(oid),
        },
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id


# Export models
__all__ = ["AtomicFactRecord", "AtomicFactRecordProjection"]
