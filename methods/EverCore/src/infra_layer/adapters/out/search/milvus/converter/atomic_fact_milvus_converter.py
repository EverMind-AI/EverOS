"""
Atomic Fact Milvus Converter

Converts MongoDB v1_atomic_fact_records to Milvus v1_atomic_fact_record.
Only maps search-essential fields for vector semantic retrieval.
"""

import json
from typing import Dict, Any

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.atomic_fact_collection import (
    AtomicFactCollection,
)
from infra_layer.adapters.out.persistence.document.memory.atomic_fact_record import (
    AtomicFactRecord as MongoAtomicFactRecord,
)
from api_specs.memory_types import RawDataType

logger = get_logger(__name__)


class AtomicFactMilvusConverter(BaseMilvusConverter[AtomicFactCollection]):
    """
    Atomic Fact Milvus Converter

    Converts MongoDB v1_atomic_fact_records documents to Milvus v1_atomic_fact_record entities.
    Only maps search-essential fields for vector semantic retrieval.
    Full data is retrieved from MongoDB using parent_id.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoAtomicFactRecord) -> Dict[str, Any]:
        """
        Convert from MongoDB v1_atomic_fact_records document to Milvus v1_atomic_fact_record entity

        Args:
            source_doc: MongoDB v1_atomic_fact_records document instance

        Returns:
            Dict[str, Any]: Milvus entity dictionary, ready for insertion
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            # Convert timestamp to integer (epoch milliseconds)
            timestamp = (
                int(source_doc.timestamp.timestamp() * 1000)
                if source_doc.timestamp
                else 0
            )

            milvus_entity = {
                # Basic identifier fields
                "id": str(source_doc.id),
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "session_id": source_doc.session_id or "",
                # Participant list
                "participants": source_doc.participants or [],
                "sender_ids": getattr(source_doc, "sender_ids", []) or [],
                # Type field
                "type": getattr(source_doc, "type", None)
                or RawDataType.CONVERSATION.value,
                # Timestamp field
                "timestamp": timestamp,
                # Parent info for MongoDB back-reference
                "parent_type": source_doc.parent_type or "",
                "parent_id": str(source_doc.parent_id) if source_doc.parent_id else "",
                # Vector field
                "vector": source_doc.vector if source_doc.vector else [],
            }

            return milvus_entity

        except Exception as e:
            logger.error(
                "Failed to convert MongoDB AtomicFact document to Milvus entity: %s", e
            )
            raise

    @staticmethod
    def _build_search_content(source_doc: MongoAtomicFactRecord) -> str:
        """Build search content (JSON list format)"""
        text_content = []

        if source_doc.atomic_fact:
            text_content.append(source_doc.atomic_fact)

        return json.dumps(text_content, ensure_ascii=False)
