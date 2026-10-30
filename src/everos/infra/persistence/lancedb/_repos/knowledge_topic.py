"""LanceDB repository for the ``knowledge_topic`` table."""

from __future__ import annotations

from everos.infra.persistence.lancedb._base.repository import LanceRepoBase

from ...records.knowledge_topic import KnowledgeTopic
from ._contract import LanceIndexContract


class _KnowledgeTopicRepo(
    LanceIndexContract[KnowledgeTopic], LanceRepoBase[KnowledgeTopic]
):
    """LanceDB repository for the ``knowledge_topic`` table."""

    schema = KnowledgeTopic
