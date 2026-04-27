"""
AgentSkill Milvus Collection Definition

AgentSkill-specific Collection class for semantic search over
reusable skills extracted from MemScene clusters.
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class AgentSkillCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    AgentSkill Milvus Collection

    Stores vector embeddings of reusable skill items.
    The vector represents the embedding of name + description.
    """

    _COLLECTION_NAME = "v1_agent_skill"

    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="AgentSkillRecord unique identifier",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTORIZE_DIMENSIONS,
                description="Embedding of name + description",
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="User ID (agent owner)",
            ),
            FieldSchema(
                name="group_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Group/session ID",
            ),
            FieldSchema(
                name="cluster_id",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="MemScene cluster ID",
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="name + newline + description (primary text field)",
            ),
            FieldSchema(
                name="maturity_score",
                dtype=DataType.FLOAT,
                description="Normalized quality score (0.0-1.0)",
            ),
            FieldSchema(
                name="confidence",
                dtype=DataType.FLOAT,
                description="Confidence score (0.0-1.0)",
            ),
        ],
        description="Vector collection for agent skill",
        enable_dynamic_field=True,
    )

    _INDEX_CONFIGS = [
        IndexConfig(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        ),
        IndexConfig(field_name="user_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="group_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="cluster_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="maturity_score", index_type="AUTOINDEX"),
        IndexConfig(field_name="confidence", index_type="AUTOINDEX"),
    ]
