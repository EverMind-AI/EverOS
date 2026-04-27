"""
V1 Episodic Memory Milvus Collection Definition

Based on MongoDB v1_episodic_memories collection.
Simplified for vector semantic retrieval - only stores search-essential fields.
Full data is retrieved from MongoDB using parent_id.
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class EpisodicMemoryCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    V1 Episodic Memory Milvus Collection

    Usage:
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Base name for the Collection
    _COLLECTION_NAME = "v1_episodic_memory"

    # Collection Schema definition
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="Event unique identifier",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTORIZE_DIMENSIONS,
                description="Text vector for semantic search",
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="User ID (None for group memory)",
            ),
            FieldSchema(
                name="group_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="Group ID",
            ),
            FieldSchema(
                name="session_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Session identifier",
            ),
            FieldSchema(
                name="participants",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=100,
                max_length=100,
                description="List of participant sender_ids",
            ),
            FieldSchema(
                name="sender_ids",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=100,
                max_length=100,
                description="Sender IDs of event participants",
            ),
            FieldSchema(
                name="type",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Episode type (e.g., Conversation, Email, etc.)",
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.INT64,
                description="Event timestamp (epoch milliseconds)",
            ),
            FieldSchema(
                name="episode",
                dtype=DataType.VARCHAR,
                max_length=10000,
                description="Episode description",
            ),
            FieldSchema(
                name="search_content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="Search content",
            ),
            FieldSchema(
                name="parent_type",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Parent memory type (e.g., memcell)",
            ),
            FieldSchema(
                name="parent_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Parent memory ID (for MongoDB back-reference)",
            ),
        ],
        description="V1 vector collection for episodic memory",
        enable_dynamic_field=True,
    )

    # Index configuration
    _INDEX_CONFIGS = [
        # Vector field index (for similarity search)
        IndexConfig(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        ),
        # Scalar field indexes (for filtering)
        IndexConfig(field_name="user_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="group_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="session_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="parent_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="timestamp", index_type="AUTOINDEX"),
    ]
