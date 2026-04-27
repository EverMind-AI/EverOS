"""
Foresight ES Converter

Converts MongoDB v1_foresight_records to ES v1_foresight_record.
"""

from typing import List
from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.elasticsearch.memory.foresight import ForesightDoc
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord as MongoForesightRecord,
)

logger = get_logger(__name__)


class ForesightConverter(BaseEsConverter[ForesightDoc]):
    """
    Foresight ES Converter

    Converts MongoDB v1 ForesightRecord documents to ES v1 ForesightDoc documents.
    Only maps search-essential fields.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoForesightRecord) -> ForesightDoc:
        """
        Convert from MongoDB v1 ForesightRecord document to ES v1 ForesightDoc instance

        Args:
            source_doc: Instance of MongoDB v1_foresight_record document

        Returns:
            ForesightDoc: ES document instance, ready for indexing
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            es_doc = ForesightDoc(
                meta={'id': str(source_doc.id)},
                # Basic identifier fields
                id=str(source_doc.id),
                user_id=source_doc.user_id,
                group_id=source_doc.group_id,
                session_id=source_doc.session_id,
                # Participant list
                participants=source_doc.participants or [],
                sender_ids=getattr(source_doc, 'sender_ids', None),
                # Core BM25 content fields
                content=source_doc.content,
                evidence=source_doc.evidence,
                search_content=getattr(source_doc, 'search_content', None),
                # Classification fields
                type=getattr(source_doc, 'type', None),
                # Parent info for MongoDB back-reference
                parent_type=source_doc.parent_type,
                parent_id=str(source_doc.parent_id) if source_doc.parent_id else None,
                # Time range fields
                start_time=getattr(source_doc, 'start_time', None),
                end_time=getattr(source_doc, 'end_time', None),
                duration_days=getattr(source_doc, 'duration_days', None),
            )

            return es_doc

        except Exception as e:
            logger.error("Failed to convert MongoDB document to ES document: %s", e)
            raise
