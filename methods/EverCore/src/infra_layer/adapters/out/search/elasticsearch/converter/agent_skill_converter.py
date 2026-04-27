"""
Agent Skill ES Converter

Converts MongoDB AgentSkillRecord to Elasticsearch AgentSkillDoc document.
"""

from typing import List
import jieba

from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.agent_skill import (
    AgentSkillDoc,
)
from infra_layer.adapters.out.persistence.document.memory.agent_skill import (
    AgentSkillRecord,
)

logger = get_logger(__name__)


class AgentSkillConverter(BaseEsConverter[AgentSkillDoc]):
    """
    Agent Skill ES Converter

    Converts MongoDB agent skill documents to Elasticsearch AgentSkillDoc documents.
    Combines name + description + content for BM25 retrieval.
    """

    @classmethod
    def from_mongo(cls, source_doc: AgentSkillRecord) -> AgentSkillDoc:
        """
        Convert from MongoDB agent skill document to ES document

        Args:
            source_doc: Instance of MongoDB agent skill document

        Returns:
            AgentSkillDoc: Instance of ES document
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            search_content = cls._build_search_content(source_doc)

            es_doc = AgentSkillDoc(
                meta={"id": str(source_doc.id)},
                id=str(source_doc.id),
                user_id=source_doc.user_id,
                group_id=source_doc.group_id,
                cluster_id=source_doc.cluster_id,
                search_content=search_content,
                name=source_doc.name or "",
                description=source_doc.description or "",
                content=source_doc.content or "",
                confidence=source_doc.confidence,
                maturity_score=source_doc.maturity_score,
            )

            return es_doc

        except Exception as e:
            logger.error("Failed to convert AgentSkillRecord to ES document: %s", e)
            raise

    @classmethod
    def _build_search_content(cls, source_doc: AgentSkillRecord) -> List[str]:
        """
        Build search content list from skill fields.

        Segments name + description + content with jieba, filters stopwords,
        and generates keyword list for BM25 retrieval.
        """
        search_content = []

        # Combine name + description + content
        text_parts = [
            source_doc.name or "",
            source_doc.description or "",
            source_doc.content or "",
        ]

        for text in text_parts:
            if text:
                words = jieba.lcut(text)
                search_content.extend(filter_stopwords(words, min_length=2))

        # Deduplicate while preserving order
        seen = set()
        unique_content = []
        for word in search_content:
            if word not in seen and word.strip():
                seen.add(word)
                unique_content.append(word)

        # Fallback: use raw description if empty after filtering
        if not unique_content and source_doc.description:
            return [source_doc.description]

        return unique_content if unique_content else [""]
