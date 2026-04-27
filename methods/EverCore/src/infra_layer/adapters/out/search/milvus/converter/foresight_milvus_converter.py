"""
Foresight Milvus Converter

Converts MongoDB v1_foresight_records to Milvus v1_foresight_record.
Only maps search-essential fields for vector semantic retrieval.
"""

import json
from typing import Dict, Any
from datetime import datetime

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.foresight_collection import (
    ForesightCollection,
)
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord as MongoForesightRecord,
)

logger = get_logger(__name__)


class ForesightMilvusConverter(BaseMilvusConverter[ForesightCollection]):
    """
    Foresight Milvus Converter

    Converts MongoDB v1_foresight_records documents to Milvus v1_foresight_record entities.
    Only maps search-essential fields for vector semantic retrieval.
    Full data is retrieved from MongoDB using parent_id.
    """

    @classmethod
    def _parse_time_field(cls, time_value, field_name: str, doc_id) -> int:
        """Parse time field to epoch milliseconds, return 0 and log warning on failure"""
        if not time_value:
            return 0

        try:
            if isinstance(time_value, datetime):
                return int(time_value.timestamp() * 1000)
            elif isinstance(time_value, str):
                dt = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            elif isinstance(time_value, (int, float)):
                return int(time_value * 1000)
        except Exception as e:
            logger.warning(
                f"Failed to parse {field_name} (doc_id={doc_id}): {time_value}, error: {e}"
            )

        return 0

    @classmethod
    def from_mongo(cls, source_doc: MongoForesightRecord) -> Dict[str, Any]:
        """
        Convert from MongoDB v1_foresight_records document to Milvus v1_foresight_record entity

        Args:
            source_doc: MongoDB v1_foresight_records document instance

        Returns:
            Dict[str, Any]: Milvus entity dictionary, ready for insertion
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            # Parse time fields
            start_time = cls._parse_time_field(
                source_doc.start_time, "start_time", source_doc.id
            )
            end_time = cls._parse_time_field(
                source_doc.end_time, "end_time", source_doc.id
            )

            # Build search content
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
                # Time fields
                "start_time": start_time,
                "end_time": end_time,
                "duration_days": (
                    source_doc.duration_days if source_doc.duration_days else 0
                ),
                # Core content fields
                "content": source_doc.content,
                "evidence": source_doc.evidence or "",
                "search_content": search_content,
                # Parent info for MongoDB back-reference
                "parent_type": source_doc.parent_type or "",
                "parent_id": str(source_doc.parent_id) if source_doc.parent_id else "",
                # Vector field
                "vector": source_doc.vector if source_doc.vector else [],
            }

            return milvus_entity

        except Exception as e:
            logger.error(
                "Failed to convert MongoDB foresight document to Milvus entity: %s", e
            )
            raise

    @staticmethod
    def _build_search_content(source_doc: MongoForesightRecord) -> str:
        """Build search content (JSON list format)"""
        text_content = []

        # Main content
        if source_doc.content:
            text_content.append(source_doc.content)

        # Add evidence to improve retrieval capability
        if source_doc.evidence:
            text_content.append(source_doc.evidence)

        return json.dumps(text_content, ensure_ascii=False)
