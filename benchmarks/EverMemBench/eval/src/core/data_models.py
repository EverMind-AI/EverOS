"""
Core data models for multi-person group chat evaluation.

Defines standard data formats for:
- GroupChatMessage: Individual message in a group chat
- GroupChatDay: All messages for a single day (organized by groups)
- Dataset: Complete dataset with multiple days
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class GroupChatMessage:
    """
    Single message in a group chat.
    
    Attributes:
        speaker: Speaker name (e.g., "Weihua Zhang")
        content: Original dialogue text
        timestamp: Parsed datetime from "time" field
        group: Group name (e.g., "Group 1")
        date: Date string (e.g., "2025-01-09")
        metadata: Additional fields (task_ids, etc.)
    """
    speaker: str
    content: str
    timestamp: datetime
    group: str
    date: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupChatDay:
    """
    All messages for a single day, organized by groups.
    
    Attributes:
        date: Date string (e.g., "2025-01-09")
        groups: Dict mapping group name to list of messages
        metadata: Additional metadata for the day
    """
    date: str
    groups: Dict[str, List[GroupChatMessage]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_messages(self) -> int:
        """Total number of messages across all groups for this day."""
        return sum(len(msgs) for msgs in self.groups.values())
    
    @property
    def group_names(self) -> List[str]:
        """List of group names for this day."""
        return list(self.groups.keys())


@dataclass
class Dataset:
    """
    Complete dataset with multiple days of group chat data.
    
    Attributes:
        name: Dataset identifier (e.g., "groupchat_004")
        days: List of GroupChatDay objects, ordered by date
        metadata: Dataset-level metadata
    """
    name: str
    days: List[GroupChatDay]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_days(self) -> int:
        """Total number of days in the dataset."""
        return len(self.days)
    
    @property
    def total_messages(self) -> int:
        """Total number of messages across all days."""
        return sum(day.total_messages for day in self.days)
    
    @property
    def date_range(self) -> tuple:
        """Return (start_date, end_date) tuple."""
        if not self.days:
            return (None, None)
        dates = [day.date for day in self.days]
        return (min(dates), max(dates))
    
    def get_day(self, date: str) -> Optional[GroupChatDay]:
        """Get GroupChatDay by date string."""
        for day in self.days:
            if day.date == date:
                return day
        return None
    
    def get_days_subset(self, num_days: int) -> "Dataset":
        """
        Return a new Dataset with only the first N days.
        
        Args:
            num_days: Number of days to include
            
        Returns:
            New Dataset with subset of days
        """
        subset_days = self.days[:num_days]
        return Dataset(
            name=f"{self.name}_subset_{num_days}",
            days=subset_days,
            metadata={
                **self.metadata,
                "is_subset": True,
                "original_days": len(self.days),
                "subset_days": len(subset_days),
            }
        )


@dataclass
class AddResult:
    """
    Result of Add stage execution.
    
    Attributes:
        success: Whether the add operation succeeded
        days_processed: Number of days processed
        messages_sent: Total messages sent
        errors: List of error messages (if any)
        metadata: Additional result metadata
    """
    success: bool
    days_processed: int
    messages_sent: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# QA Evaluation Data Models
# ============================================================================

@dataclass
class QAItem:
    """
    Single QA item for evaluation.

    Supports two question types:
    - multiple_choice: Has options field, answer is a letter (A/B/C/D)
    - open_ended: No options, answer is free text

    Attributes:
        question_id: Unique identifier for the question
        question: The question text
        answer: Golden answer (correct answer text)
        question_type: "multiple_choice" or "open_ended"
        options: List of options for multiple choice (e.g., ["A. Option1", "B. Option2"])
        correct_option: Correct option letter for multiple choice (e.g., "A")
        metadata: Additional fields
    """
    question_id: str
    question: str
    answer: str
    question_type: str  # "multiple_choice" or "open_ended"
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    Result of memory search for a single question.
    
    Attributes:
        question_id: ID of the question this search is for
        query: The search query (usually the question text)
        retrieved_memories: List of retrieved memory strings
        context: Formatted context string for LLM prompt
        search_duration_ms: Time taken for search in milliseconds
        metadata: Additional search metadata (scores, etc.)
    """
    question_id: str
    query: str
    retrieved_memories: List[str]
    context: str
    search_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    """
    Result of answer generation for a single question.
    
    Attributes:
        question_id: ID of the question
        question: The question text
        question_type: "multiple_choice" or "open_ended"
        golden_answer: The correct/expected answer
        generated_answer: The model-generated answer
        search_result: Associated search result
        answer_duration_ms: Time taken for answer generation
        metadata: Additional metadata
    """
    question_id: str
    question: str
    question_type: str
    golden_answer: str
    generated_answer: str
    search_result: SearchResult
    answer_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LightAnswerResult:
    """
    Lightweight answer result for LLM system (memory-efficient).
    
    Used when full dialogue context is shared across all questions,
    avoiding redundant storage of SearchResult/context in each result.
    This significantly reduces memory usage for large dialogues.
    
    Attributes:
        question_id: ID of the question
        question: The question text
        question_type: "multiple_choice" or "open_ended"
        golden_answer: The correct/expected answer
        generated_answer: The model-generated answer
        answer_duration_ms: Time taken for answer generation
    """
    question_id: str
    question: str
    question_type: str
    golden_answer: str
    generated_answer: str
    answer_duration_ms: float = 0.0


@dataclass
class EvaluationResult:
    """
    Result of evaluation for all questions.
    
    Attributes:
        total_questions: Total number of questions evaluated
        correct: Number of correct answers
        accuracy: Overall accuracy (correct / total)
        accuracy_by_type: Accuracy broken down by question type
        detailed_results: List of per-question evaluation results
        metadata: Additional evaluation metadata
    """
    total_questions: int
    correct: int
    accuracy: float
    accuracy_by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

