"""
AgentCaseRecord - Beanie ODM model for agent cases.

Stores a compressed agent task-solving experience extracted from an agent conversation MemCell.
Each record has: task_intent, approach, quality_score.
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


class AgentCaseRecord(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    Agent case document model.

    Stores the compressed representation of one agent task-solving interaction.
    One MemCell produces at most one AgentCaseRecord.
    """

    # Identity fields
    user_id: Optional[str] = Field(
        default=None, description="User ID who initiated the task"
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(..., description="Task occurrence time")

    # Core experience fields (flat, one experience per record)
    task_intent: str = Field(
        default="", description="Rewritten task intent as retrieval key"
    )
    approach: str = Field(
        default="", description="Step-by-step approach with decisions and lessons"
    )
    quality_score: Optional[float] = Field(
        default=None, description="Task completion quality score (0.0-1.0)"
    )
    key_insight: Optional[str] = Field(
        default=None, description="Pivotal strategy shift or decision"
    )
    # Parent linkage (to MemCell)
    parent_type: Optional[str] = Field(
        default=None, description="Parent memory type (e.g., memcell)"
    )
    parent_id: Optional[str] = Field(
        default=None, description="Parent memory ID (MemCell event_id)"
    )

    # Vector embedding
    vector: Optional[List[float]] = Field(
        default=None, description="Embedding vector of task_intent"
    )
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    model_config = ConfigDict(
        collection="v1_agent_cases",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "session_abc",
                "timestamp": "2026-02-14T10:30:00.000Z",
                "task_intent": "Search for open source Python web frameworks and compare their GitHub stars",
                "approach": "1. Searched GitHub for Python web frameworks with >5K stars using web_search\n2. Selected top 3: Django, Flask, FastAPI\n3. Compared GitHub stars and activity metrics\n   - Result: FastAPI has fastest growth rate",
                "quality_score": 0.85,
                "parent_type": "memcell",
                "parent_id": "67af1234abcd5678ef901234",
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "v1_agent_cases"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("deleted_at", ASCENDING)],
                name="idx_tenant_deleted_at",
                sparse=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("parent_id", ASCENDING)],
                name="idx_tenant_parent_id",
                sparse=True,
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


class AgentCaseProjection(DocumentBase, AuditBase):
    """
    Simplified agent case model (without vector)

    Used in GET queries where vector data is not needed,
    reducing data transfer and memory usage.
    """

    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    timestamp: datetime = Field(..., description="Task occurrence time")
    task_intent: str = Field(default="", description="Rewritten task intent")
    approach: str = Field(default="", description="Step-by-step approach")
    quality_score: Optional[float] = Field(
        default=None, description="Task completion quality score (0.0-1.0)"
    )
    key_insight: Optional[str] = Field(
        default=None, description="Pivotal strategy shift or decision"
    )
    parent_type: Optional[str] = Field(default=None, description="Parent memory type")
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")
