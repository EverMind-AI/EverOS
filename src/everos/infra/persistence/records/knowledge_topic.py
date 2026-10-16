"""LanceDB table schema for L2 knowledge topic nodes."""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

from ._dim import configured_dim

_DIM = configured_dim()


class KnowledgeTopic(BaseLanceTable):
    """L2 topic node — dense + dual-column BM25 retrieval."""

    TABLE_NAME: ClassVar[str] = "knowledge_topic"
    BM25_FIELDS: ClassVar[list[str]] = ["summary_tokens", "content_tokens"]
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")

    id: str
    doc_id: str
    category_id: str
    app_id: str
    project_id: str
    topic_name: str
    topic_path: str
    depth: int
    parent_node_id: str = ""
    summary: str
    summary_tokens: str
    content_tokens: str
    content_labels: list[str] = []
    md_path: str
    content_sha256: str
    vector: Vector(_DIM) | None = None  # type: ignore[valid-type]  -- Vector() is runtime-constructed; static analyzers cannot verify
    created_at: dt.datetime
    updated_at: dt.datetime
