"""everalgo knowledge types <-> domain knowledge models."""

from __future__ import annotations

__all__ = ["category_to_algo", "memory_from_algo"]

from everalgo.types import CategorySpec
from everalgo.types import KnowledgeMemory as AlgoKnowledgeMemory

from everos.memory.models import KnowledgeCategory, KnowledgeMemory


def memory_from_algo(algo: AlgoKnowledgeMemory) -> KnowledgeMemory:
    """One extracted topic-tree node as a domain knowledge memory."""
    return KnowledgeMemory.model_validate(algo.model_dump())


def category_to_algo(category: KnowledgeCategory) -> CategorySpec:
    """One taxonomy entry as the category spec the extractor classifies into."""
    return CategorySpec(id=category.id, description=category.description)
