"""Frontmatter schema for ``knowledge/{category}/{doc_title}/<n>_<name>.md``."""

from __future__ import annotations

from typing import Literal

from everos.core.persistence.markdown import (
    BaseFrontmatter,
    KnowledgeScopedMixin,
    KnowledgeTopicPathMixin,
)


class KnowledgeTopicFrontmatter(
    KnowledgeTopicPathMixin,
    KnowledgeScopedMixin,
    BaseFrontmatter,
):
    """L2 topic-node frontmatter. Body = topic content full text."""

    type: Literal["knowledge_topic"] = "knowledge_topic"
    node_id: str
    doc_id: str
    category_id: str
    topic_index: int
    topic_name: str
    topic_path: str
    summary: str
    depth: int
    parent_node_id: str | None = None
    children_node_ids: list[str] = []
    content_labels: list[str] = []
