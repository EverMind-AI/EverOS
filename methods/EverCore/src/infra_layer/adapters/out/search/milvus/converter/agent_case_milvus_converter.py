"""
AgentCase Milvus Converter

Converts MongoDB AgentCaseRecord documents into Milvus Collection entities.
"""

from typing import Dict, Any

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.agent_case_collection import (
    AgentCaseCollection,
)
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseRecord,
)

logger = get_logger(__name__)


class AgentCaseMilvusConverter(BaseMilvusConverter[AgentCaseCollection]):
    """
    Converts MongoDB AgentCaseRecord documents into Milvus entities.

    Vector field: embedding of task_intent.
    task_intent field: task intent string for direct text access and search.
    """

    @classmethod
    def from_mongo(cls, source_doc: AgentCaseRecord) -> Dict[str, Any]:
        """
        Convert from MongoDB AgentCaseRecord to Milvus entity dict.

        Args:
            source_doc: MongoDB AgentCaseRecord document instance

        Returns:
            Dict[str, Any]: Milvus entity dictionary ready for insertion

        Raises:
            ValueError: If source_doc is None
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            task_intent = source_doc.task_intent or ""

            # Timestamps
            timestamp = (
                int(source_doc.timestamp.timestamp()) if source_doc.timestamp else 0
            )

            entity = {
                "id": str(source_doc.id),
                "vector": source_doc.vector if source_doc.vector else [],
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "session_id": source_doc.session_id or "",
                "timestamp": timestamp,
                "task_intent": task_intent[:5000],
                "parent_type": source_doc.parent_type or "",
                "parent_id": source_doc.parent_id or "",
            }

            return entity

        except Exception as e:
            logger.error("Failed to convert AgentCaseRecord to Milvus entity: %s", e)
            raise
