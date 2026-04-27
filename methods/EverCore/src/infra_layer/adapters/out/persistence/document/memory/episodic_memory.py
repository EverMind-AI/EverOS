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


class EpisodicMemory(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    Episodic memory document model

    Stores user's episodic memories, including event summaries, participants, topics, etc.
    Directly transferred from MemCell summaries.
    """

    # field from api input
    user_id: Optional[str] = Field(
        default=None, description="The individual involved, None indicates group memory"
    )
    # field from api input
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(..., description="Occurrence time (timestamp)")
    participants: Optional[List[str]] = Field(
        default=None, description="IDs of event participants"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of messages"
    )
    summary: str = Field(..., min_length=1, description="Memory unit")
    subject: Optional[str] = Field(default=None, description="Memory unit subject")
    episode: str = Field(..., min_length=1, description="Episodic memory")
    type: Optional[str] = Field(
        default=None, description="Episode type, such as Conversation"
    )

    parent_type: Optional[str] = Field(
        default=None, description="Parent memory type (e.g., memcell)"
    )
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")

    vector: Optional[List[float]] = Field(default=None, description="Text vector")
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    model_config = ConfigDict(
        collection="v1_episodic_memories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "group_work",
                "timestamp": 1701388800,
                "participants": ["Zhang San", "Li Si"],
                "summary": "Discussed project progress and next week's plan",
                "subject": "Project meeting",
                "episode": "Held a project progress discussion in the meeting room, confirmed next week's development task assignments",
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

        name = "v1_episodic_memories"
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


class EpisodicMemoryProjection(DocumentBase, AuditBase):
    """
    Simplified episodic memory model (without vector)

    Used in most scenarios where vector data is not needed, reducing data transfer and memory usage.
    """

    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    user_id: Optional[str] = Field(
        default=None, description="The individual involved, None indicates group memory"
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(..., description="Occurrence time (timestamp)")
    participants: Optional[List[str]] = Field(
        default=None, description="Names of event participants"
    )
    sender_ids: Optional[List[str]] = Field(
        default=None, description="Sender IDs of event participants"
    )
    summary: str = Field(..., min_length=1, description="Memory unit")
    subject: Optional[str] = Field(default=None, description="Memory unit subject")
    episode: str = Field(..., min_length=1, description="Episodic memory")
    type: Optional[str] = Field(
        default=None, description="Episode type, such as Conversation"
    )

    parent_type: Optional[str] = Field(
        default=None, description="Parent memory type (e.g., memcell)"
    )
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")

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
        return self.id


# Export models
__all__ = ["EpisodicMemory", "EpisodicMemoryProjection"]
