"""
V1 User Profile Milvus Collection Definition

Based on MongoDB v1_user_profiles collection.
Simplified for vector semantic retrieval of user profiles.

Note: UserProfile does NOT have session_id (user-level aggregation).
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class UserProfileCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    V1 User Profile Milvus Collection

    Usage:
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Base name for the Collection
    _COLLECTION_NAME = "v1_user_profile"

    # Collection Schema definition
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="Profile unique identifier",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTORIZE_DIMENSIONS,
                description="Profile data vector for semantic search",
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="User ID (required)",
            ),
            FieldSchema(
                name="group_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Group ID",
            ),
            FieldSchema(
                name="scenario",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Scenario type: solo or team",
            ),
            FieldSchema(
                name="memcell_count",
                dtype=DataType.INT64,
                description="Number of MemCells involved in extraction",
            ),
            FieldSchema(
                name="item_type",
                dtype=DataType.VARCHAR,
                max_length=32,
                description="Item type: explicit_info or implicit_trait",
            ),
            FieldSchema(
                name="embed_text",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="Text used for generating embedding vector",
            ),
        ],
        description="V1 vector collection for user profiles",
        enable_dynamic_field=True,
    )

    # Index configuration
    _INDEX_CONFIGS = [
        # Vector field index
        IndexConfig(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        ),
        # Scalar field indexes
        IndexConfig(field_name="user_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="group_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="scenario", index_type="AUTOINDEX"),
    ]
