"""LanceDB repo singleton for the ``knowledge_topic`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceRepoBase

from ..lancedb_manager import get_table
from ..tables.knowledge_topic import KnowledgeTopic


class _KnowledgeTopicRepo(LanceRepoBase[KnowledgeTopic]):
    """LanceDB repository for the ``knowledge_topic`` table."""

    schema = KnowledgeTopic

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


knowledge_topic_repo = _KnowledgeTopicRepo()
