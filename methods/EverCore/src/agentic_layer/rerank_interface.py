"""
Rerank Service Interface

Defines the abstract interface for all reranking service implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from api_specs.memory_models import MemoryType


class RerankError(Exception):
    """Rerank API error exception class"""


@dataclass
class RerankMemResponse:
    """Reranked memory retrieval response"""

    memories: List[Dict[str, List[Any]]] = field(default_factory=list)
    scores: List[Dict[str, List[float]]] = field(default_factory=list)
    rerank_scores: List[Dict[str, List[float]]] = field(default_factory=list)
    original_data: List[Dict[str, List[Dict[str, Any]]]] = field(default_factory=list)
    total_count: int = 0
    has_more: bool = False
    query_metadata: Any = field(default_factory=dict)
    metadata: Any = field(default_factory=dict)


class RerankServiceInterface(ABC):
    """Reranking service interface"""

    @abstractmethod
    async def rerank_memories(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank memories based on query

        Args:
            query: Query text
            hits: List of memory hits to rerank (each hit is a dict with memory data)
            top_k: Return top K results (optional, if None returns all reranked results)
            instruction: Optional reranking instruction

        Returns:
            List of reranked memory hits, sorted by relevance score
        """
        ...

    @abstractmethod
    async def rerank_documents(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rerank raw documents (low-level API)

        Args:
            query: Query text
            documents: List of document strings to rerank
            instruction: Optional reranking instruction

        Returns:
            Dict with 'results' key containing list of {index, score, rank}
        """
        ...

    @abstractmethod
    async def close(self):
        """Close and cleanup resources"""
        ...


def extract_text_from_hit(hit: Dict[str, Any]) -> str:
    """
    Extract text content from a memory hit for reranking.

    Concatenates relevant text fields based on memory_type.
    Used by rerank implementations and the hybrid service for token counting.

    Args:
        hit: Memory hit dict, optionally with '_source' nested data

    Returns:
        Extracted text string for reranking
    """
    source = hit.get('_source', hit)
    memory_type = hit.get('memory_type', '')

    match memory_type:
        case MemoryType.EPISODIC_MEMORY.value:
            episode = source.get('episode', '')
            if episode:
                return f"Episode Memory: {episode}"
        case MemoryType.FORESIGHT.value:
            foresight = source.get('foresight', '') or source.get('content', '')
            evidence = source.get('evidence', '')
            if foresight:
                if evidence:
                    return f"Foresight: {foresight} (Evidence: {evidence})"
                return f"Foresight: {foresight}"
        case MemoryType.ATOMIC_FACT.value:
            atomic_fact = source.get('atomic_fact', '')
            if atomic_fact:
                return f"Atomic Fact: {atomic_fact}"
        case MemoryType.AGENT_CASE.value:
            task_intent = source.get('task_intent', '')
            if task_intent:
                return f"Agent Case: {task_intent}"
        case MemoryType.AGENT_SKILL.value:
            name = source.get('name', '')
            description = source.get('description', '')
            if name:
                if description:
                    return f"Agent Skill: {name} - {description}"
                return f"Agent Skill: {name}"

    if source.get('episode'):
        return source['episode']
    if source.get('atomic_fact'):
        return source['atomic_fact']
    if source.get('foresight'):
        return source['foresight']
    if source.get('content'):
        return source['content']
    if source.get('summary'):
        return source['summary']
    if source.get('subject'):
        return source['subject']
    return str(hit)
