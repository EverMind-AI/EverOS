"""
EpisodicMemory ES Converter

Responsible for converting MongoDB v1_episodic_memories to ES v1_episodic_memory.
"""

from typing import List
import jieba
from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
    EpisodicMemoryDoc,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory as MongoEpisodicMemory,
)

logger = get_logger(__name__)


class EpisodicMemoryConverter(BaseEsConverter[EpisodicMemoryDoc]):
    """
    EpisodicMemory Converter

    Converts MongoDB v1 EpisodicMemory documents to ES v1 EpisodicMemoryDoc documents.
    Only maps search-essential fields.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEpisodicMemory) -> EpisodicMemoryDoc:
        """
        Convert from MongoDB v1 EpisodicMemory document to ES v1 EpisodicMemoryDoc instance

        Args:
            source_doc: Instance of MongoDB v1_episodic_memory document

        Returns:
            EpisodicMemoryDoc: ES document instance, ready for indexing
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            es_doc = EpisodicMemoryDoc(
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
                # Core BM25 content fields
                summary=source_doc.summary,
                subject=source_doc.subject,
                episode=source_doc.episode,
                search_content=cls._build_search_content(source_doc),
                # Classification fields
                type=source_doc.type,
                # Parent info for MongoDB back-reference
                parent_type=source_doc.parent_type,
                parent_id=str(source_doc.parent_id) if source_doc.parent_id else None,
            )

            return es_doc

        except Exception as e:
            logger.error("Failed to convert MongoDB document to ES document: %s", e)
            raise

    @classmethod
    def _build_search_content(cls, source_doc: MongoEpisodicMemory) -> List[str]:
        """
        Build search content list for BM25 retrieval

        Combines multiple text fields from the MongoDB document and processes them
        with jieba word segmentation, generating a list of search content for BM25 retrieval.

        Args:
            source_doc: Instance of MongoDB's EpisodicMemory document

        Returns:
            List[str]: List of search content after jieba word segmentation
        """
        text_content = []

        # Collect all text content - including subject, summary, episode
        if hasattr(source_doc, 'subject') and source_doc.subject:
            text_content.append(source_doc.subject)

        if hasattr(source_doc, 'summary') and source_doc.summary:
            text_content.append(source_doc.summary)

        if hasattr(source_doc, 'episode') and source_doc.episode:
            text_content.append(source_doc.episode)

        # Combine all text content and apply jieba word segmentation
        combined_text = ' '.join(text_content)
        search_content = list(jieba.cut(combined_text))

        # Filter out empty strings and stopwords
        query_words = filter_stopwords(search_content, min_length=2)

        search_content = [word.strip() for word in query_words if word.strip()]

        return search_content
