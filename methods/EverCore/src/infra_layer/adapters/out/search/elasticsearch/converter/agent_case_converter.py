"""
Agent Case ES Converter

Converts MongoDB AgentCaseRecord to Elasticsearch AgentCaseDoc document.
"""

from typing import List
import jieba

from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.agent_case import AgentCaseDoc
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseRecord,
)

logger = get_logger(__name__)


class AgentCaseConverter(BaseEsConverter[AgentCaseDoc]):
    """
    Agent Case ES Converter

    Converts MongoDB agent case documents to Elasticsearch AgentCaseDoc documents.
    Extracts task_intent and approach for BM25 retrieval.
    """

    @classmethod
    def from_mongo(cls, source_doc: AgentCaseRecord) -> AgentCaseDoc:
        """
        Convert from MongoDB agent case document to ES document

        Args:
            source_doc: Instance of MongoDB agent case document

        Returns:
            AgentCaseDoc: Instance of ES document
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            search_content = cls._build_search_content(source_doc)

            es_doc = AgentCaseDoc(
                meta={"id": str(source_doc.id)},
                id=str(source_doc.id),
                user_id=source_doc.user_id,
                group_id=source_doc.group_id,
                session_id=source_doc.session_id,
                timestamp=source_doc.timestamp,
                search_content=search_content,
                task_intent=source_doc.task_intent or "",
                approach=source_doc.approach or "",
                parent_type=source_doc.parent_type,
                parent_id=source_doc.parent_id,
            )

            return es_doc

        except Exception as e:
            logger.error("Failed to convert AgentCaseRecord to ES document: %s", e)
            raise

    @classmethod
    def _build_search_content(cls, source_doc: AgentCaseRecord) -> List[str]:
        """
        Build search content list from experience fields.

        Segments task_intent and approach with jieba, filters stopwords,
        and generates keyword list for BM25 retrieval.
        """
        search_content = []

        task_intent = source_doc.task_intent or ""
        if task_intent:
            words = jieba.lcut(task_intent)
            search_content.extend(filter_stopwords(words, min_length=2))

        approach = source_doc.approach or ""
        if approach:
            words = jieba.lcut(approach)
            search_content.extend(filter_stopwords(words, min_length=2))

        # Deduplicate while preserving order
        seen = set()
        unique_content = []
        for word in search_content:
            if word not in seen and word.strip():
                seen.add(word)
                unique_content.append(word)

        # Fallback: use raw task_intent if empty after filtering
        if not unique_content:
            return [task_intent] if task_intent else [""]

        return unique_content
