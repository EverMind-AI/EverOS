from datetime import datetime
from typing import List, Optional, Dict, Any
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBase,
)
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase
from pymongo import IndexModel, ASCENDING


class MemScene(TenantAwareDocumentBase, AuditBase):
    """
    MemScene document — stores incremental clustering state for a group.
    Used for memcell-to-cluster assignment and profile extraction scheduling.
    """

    group_id: Optional[str] = Field(default=None, description="Group ID")

    # Per-memcell cluster assignment: { event_id: { memscene: cluster_id, timestamp: epoch_seconds } }
    # Used to look up which cluster a memcell belongs to, and to fetch all memcells in a cluster.
    memcell_info: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-memcell cluster assignment and timestamp"
    )

    # Per-cluster aggregated state: { cluster_id: { center: [vector], timestamp: epoch_seconds, count: int } }
    # center for similarity matching, timestamp for temporal gating and profile extraction scheduling.
    memscene_info: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-cluster centroid, latest timestamp, and member count",
    )

    # Cluster IDs that contain agent conversation (case) memcells.
    # Used to route case memcells to LLM-based clustering and exclude them from embedding-only clustering.
    case_cluster_ids: Optional[List[str]] = Field(
        default_factory=list, description="Cluster IDs containing agent case memcells"
    )

    # Auto-increment counter for cluster ID generation (cluster_000, cluster_001, ...).
    next_cluster_idx: int = Field(
        default=0, description="Counter for generating unique cluster IDs"
    )

    class Settings:
        name = "v1_mem_scenes"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("group_id", ASCENDING)],
                name="idx_tenant_group_id",
            )
        ]
