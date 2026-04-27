"""
AgentSkillRecord - Beanie ODM model for agent skill.

Stores reusable skills extracted from clustered AgentCases
within a MemScene (cluster).
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


class AgentSkillRecord(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    Agent skill document model.

    Stores a single reusable skill extracted from a MemScene
    (cluster of semantically similar AgentCase records).

    Skills are derived by analyzing patterns across multiple AgentCases
    in the same cluster, then merging/refining on each subsequent experience.
    """

    # Cluster linkage (MemScene)
    cluster_id: str = Field(
        ..., description="MemScene cluster ID this skill belongs to"
    )

    # Identity fields
    user_id: Optional[str] = Field(default=None, description="User ID (agent owner)")
    group_id: Optional[str] = Field(default=None, description="Group ID")

    # Core content
    name: Optional[str] = Field(default=None, description="Skill name")
    description: Optional[str] = Field(
        default=None,
        description="A clear description of what this skill does and when to use it",
    )
    content: str = Field(..., description="Full skill content")

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0), increases with more supporting experiences",
    )

    # Maturity assessment
    maturity_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Normalized quality score (0.0-1.0), skill is retrievable when >= threshold",
    )

    # Vector embedding for semantic retrieval
    vector: Optional[List[float]] = Field(
        default=None, description="Embedding vector of name + description"
    )
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    # Source traceability: AgentCase IDs that triggered add/update of this skill.
    source_case_ids: List[str] = Field(
        default_factory=list,
        description="AgentCase IDs that triggered add/update of this skill",
    )

    model_config = ConfigDict(
        collection="v1_agent_skills",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "cluster_id": "cluster_001",
                "name": "Technical comparison research",
                "description": "Compare open source technical solutions or frameworks by searching, extracting, and evaluating key metrics",
                "content": "1. search(tech + open source + github)\n2. Extract repo list from results\n3. Open README for each repo\n4. Compare by stars, activity, and features",
                "confidence": 0.85,
            }
        },
        extra="allow",
    )

    @property
    def skill_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "v1_agent_skills"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("deleted_at", ASCENDING)],
                name="idx_tenant_deleted_at",
                sparse=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("cluster_id", ASCENDING)],
                name="idx_tenant_cluster_id",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("group_id", ASCENDING),
                    ("cluster_id", ASCENDING),
                ],
                name="idx_tenant_group_cluster",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("user_id", ASCENDING)],
                name="idx_tenant_user_id",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("maturity_score", ASCENDING)],
                name="idx_tenant_maturity_score",
                sparse=True,
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("confidence", ASCENDING)],
                name="idx_tenant_confidence",
                sparse=True,
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


class AgentSkillProjection(DocumentBase, AuditBase):
    """
    Simplified agent skill model (without vector)

    Used in GET queries where vector data is not needed,
    reducing data transfer and memory usage.
    """

    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    cluster_id: str = Field(..., description="MemScene cluster ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    name: Optional[str] = Field(default=None, description="Skill name")
    description: Optional[str] = Field(default=None, description="Skill description")
    content: str = Field(..., description="Full skill content")
    confidence: float = Field(default=0.0, description="Confidence score (0.0-1.0)")
    maturity_score: float = Field(default=0.6, description="Maturity score (0.0-1.0)")
    source_case_ids: List[str] = Field(
        default_factory=list, description="AgentCase IDs"
    )
