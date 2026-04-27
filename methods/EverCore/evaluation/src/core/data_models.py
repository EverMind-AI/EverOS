"""
Core data models.

Define standard data formats for the evaluation framework to ensure interoperability
between different systems and datasets.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Message:
    """Standard message format."""

    sender_id: str
    sender_name: str
    content: str
    timestamp: Optional[datetime] = (
        None  # Optional, some datasets (e.g., PersonaMem) lack timestamps
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Standard conversation format."""

    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPair:
    """
    Standard QA pair format.

    Note: category field is unified as string type to be compatible with different datasets.
    """

    question_id: str
    question: str
    answer: str
    category: Optional[str] = None  # Unified as string type
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Standard dataset format."""

    dataset_name: str
    conversations: List[Conversation]
    qa_pairs: List[QAPair]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Standard search result format."""

    query: str
    conversation_id: str
    results: List[
        Dict[str, Any]
    ]  # [{"content": str, "score": float, "metadata": dict}]
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    """Standard answer result format."""

    question_id: str
    question: str
    answer: str
    golden_answer: str
    category: Optional[int] = None
    conversation_id: str = ""
    formatted_context: str = ""  # Actual context used
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Standard evaluation result format."""

    total_questions: int
    correct: int
    accuracy: float
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
