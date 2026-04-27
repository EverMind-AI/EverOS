"""
AtomicFact ES Converter

Converts MongoDB v1_atomic_fact_records to ES v1_atomic_fact_record.
"""

from typing import List
import jieba
from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.atomic_fact import (
    AtomicFactDoc,
)
from infra_layer.adapters.out.persistence.document.memory.atomic_fact_record import (
    AtomicFactRecord as MongoAtomicFactRecord,
)

logger = get_logger(__name__)


class AtomicFactConverter(BaseEsConverter[AtomicFactDoc]):
    """
    Atomic Fact ES Converter

    Converts MongoDB v1 AtomicFactRecord documents to ES v1 AtomicFactDoc documents.
    Only maps search-essential fields.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoAtomicFactRecord) -> AtomicFactDoc:
        """
        Convert from MongoDB v1 AtomicFactRecord document to ES v1 AtomicFactDoc instance

        Args:
            source_doc: Instance of MongoDB v1_atomic_fact_record document

        Returns:
            AtomicFactDoc: ES document instance, ready for indexing
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            es_doc = AtomicFactDoc(
                meta={'id': str(source_doc.id)},
                # Basic identifier fields
                id=str(source_doc.id),
                user_id=source_doc.user_id,
                group_id=source_doc.group_id,
                session_id=source_doc.session_id,
                # Timestamp field
                timestamp=source_doc.timestamp,
                # Participant list
                participants=source_doc.participants or [],
                sender_ids=getattr(source_doc, 'sender_ids', None),
                # Core BM25 content field
                atomic_fact=source_doc.atomic_fact,
                search_content=getattr(source_doc, 'search_content', None),
                # Classification fields
                type=getattr(source_doc, 'type', None),
                # Parent info for MongoDB back-reference
                parent_type=source_doc.parent_type,
                parent_id=str(source_doc.parent_id) if source_doc.parent_id else None,
            )

            return es_doc

        except Exception as e:
            logger.error("Failed to convert MongoDB document to ES document: %s", e)
            raise
