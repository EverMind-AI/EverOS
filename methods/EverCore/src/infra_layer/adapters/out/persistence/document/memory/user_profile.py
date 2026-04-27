from datetime import datetime
from typing import List, Optional, Dict, Any
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBaseWithSoftDelete,
)
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase
from pymongo import IndexModel, ASCENDING, DESCENDING
from api_specs.memory_types import ScenarioType


class UserProfile(TenantAwareDocumentBaseWithSoftDelete, AuditBase):
    """
    User profile document — one per (user_id, group_id) pair.
    Automatically extracted from conversation memcells by ProfileManager.
    """

    # Primary key: (user_id, group_id) uniquely identifies a profile.
    user_id: str = Field(..., description="User ID")
    group_id: Optional[str] = Field(default=None, description="Group ID")

    # Extracted profile content (role, skills, preferences, personality, etc.).
    profile_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted profile data as JSON, structure varies by scenario",
    )

    # "solo" = 1 user + N agents, "team" = multi-user + agents.
    scenario: str = Field(
        default=ScenarioType.SOLO.value, description="Extraction scenario: solo or team"
    )

    # Incremented on each re-extraction. For debugging/monitoring profile update frequency.
    update_count: int = Field(default=1, description="Debug: profile update count")

    # LLM discriminator score (0.0–1.0). Below profile_min_confidence threshold is filtered out.
    confidence: float = Field(
        default=0.0, description="LLM confidence score for this profile"
    )

    # Number of memcells fed to the LLM in the most recent extraction.
    memcell_count: int = Field(
        default=0, description="Memcell count used in last extraction"
    )

    # Epoch seconds of the latest memcell in the last extraction.
    # Compared against memscene_info[cluster_id].timestamp to find clusters with new data.
    last_updated_ts: Optional[float] = Field(
        default=None,
        description="Latest memcell timestamp (epoch) from last extraction, for scheduling",
    )

    class Settings:
        """Beanie settings"""

        name = "v1_user_profiles"
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
                    ("group_id", ASCENDING),
                ],
                name="idx_tenant_user_group",
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
