"""
Episodic Memory Milvus Converter

Converts MongoDB v1_episodic_memories to Milvus v1_episodic_memory.
Only maps search-essential fields for vector semantic retrieval.
"""

import json
from typing import Dict, Any

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory as MongoEpisodicMemory,
)

logger = get_logger(__name__)


class EpisodicMemoryMilvusConverter(BaseMilvusConverter[EpisodicMemoryCollection]):
    """
    Episodic Memory Milvus Converter

    Converts MongoDB v1_episodic_memories documents to Milvus v1_episodic_memory entities.
    Only maps search-essential fields for vector semantic retrieval.
    Full data is retrieved from MongoDB using parent_id.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEpisodicMemory) -> Dict[str, Any]:
        """
        Convert from MongoDB v1_episodic_memories document to Milvus v1_episodic_memory entity

        Args:
            source_doc: MongoDB v1_episodic_memories document instance

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

            # Build search content and metadata
            search_content = cls._build_search_content(source_doc)

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
                "type": getattr(source_doc, "type", None) or "",
                # Timestamp field
                "timestamp": timestamp,
                # Core content fields
                "episode": source_doc.episode or "",
                "search_content": search_content,
                # Parent info for MongoDB back-reference
                "parent_type": source_doc.parent_type or "",
                "parent_id": str(source_doc.parent_id) if source_doc.parent_id else "",
                # Vector field - needs to be set externally
                "vector": (
                    source_doc.vector
                    if hasattr(source_doc, "vector") and source_doc.vector
                    else []
                ),
            }

            return milvus_entity

        except Exception as e:
            logger.error("Failed to convert MongoDB document to Milvus entity: %s", e)
            raise

    @staticmethod
    def _build_search_content(source_doc: MongoEpisodicMemory) -> str:
        """
        Build search content

        Combine key text content from the document into a search content list, return as JSON string.

        Args:
            source_doc: MongoDB EpisodicMemory document instance

        Returns:
            str: Search content JSON string (list format)
        """
        text_content = []

        # Collect all text content (by priority: subject -> summary -> content)
        if hasattr(source_doc, "subject") and source_doc.subject:
            text_content.append(source_doc.subject)

        if hasattr(source_doc, "summary") and source_doc.summary:
            text_content.append(source_doc.summary)

        if hasattr(source_doc, "episode") and source_doc.episode:
            # episode might be very long, only take first 500 characters
            text_content.append(source_doc.episode)

        # Return JSON string list format, keep consistent with MemCell synchronization logic
        return json.dumps(text_content, ensure_ascii=False)
