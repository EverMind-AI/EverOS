"""File-based knowledge extraction — parsed documents to topic-tree memories."""

from __future__ import annotations

import logging

from everos.algo.knowledge._classify import aclassify_category
from everos.algo.knowledge.extractor import KnowledgeExtractor

__all__ = [
    "KnowledgeExtractor",
    "aclassify_category",
]

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())
