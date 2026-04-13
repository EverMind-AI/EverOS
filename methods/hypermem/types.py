"""
Core data types for HyperMem hypergraph memory system.

- Episode: Memory episode for storing conversation segments
- Fact: Knowledge fact extracted from topics
- Topic: Topic abstraction grouping related episodes
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import datetime
import logging

from hypermem.utils.datetime_utils import to_iso_format

logger = logging.getLogger(__name__)


class RawDataType(Enum):
    """Types of content that can be processed."""
    CONVERSATION = "Conversation"


@dataclass
class Episode:
    """Episode representing a conversation segment."""
    event_id: str
    user_id_list: List[str]
    original_data: List[Dict[str, Any]]
    timestamp: datetime.datetime
    summary: str

    participants: Optional[List[str]] = None
    type: Optional[RawDataType] = None
    keywords: Optional[List[str]] = None
    subject: Optional[str] = None
    episode_description: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.original_data:
            raise ValueError("original_data is required")
        if not self.summary:
            raise ValueError("summary is required")

    def __repr__(self) -> str:
        return f"Episode(event_id={self.event_id}, original_data={self.original_data}, timestamp={self.timestamp}, summary={self.summary})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": to_iso_format(self.timestamp),
            "summary": self.summary,
            "participants": self.participants,
            "type": str(self.type.value) if self.type else None,
            "keywords": self.keywords,
            "subject": self.subject,
            "episode_description": self.episode_description,
        }


@dataclass
class Fact:
    """
    Fact - Core information unit extracted from topics.

    Each fact is semantically complete and can directly answer user queries.
    """
    fact_id: str
    content: str
    episode_ids: List[str]
    topic_id: str

    confidence: float = 0.8
    temporal: Optional[str] = None
    spatial: Optional[str] = None
    keywords: Optional[List[str]] = None
    query_patterns: Optional[List[str]] = None
    timestamp: Optional[datetime.datetime] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.query_patterns is None:
            self.query_patterns = []

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Fact(id={self.fact_id}, content='{content_preview}')"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "episode_ids": self.episode_ids,
            "topic_id": self.topic_id,
            "confidence": self.confidence,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "keywords": self.keywords,
            "query_patterns": self.query_patterns,
            "timestamp": to_iso_format(self.timestamp) if self.timestamp else None
        }

    def to_text(self) -> str:
        parts = [self.content]
        if self.temporal:
            parts.append(f"Time: {self.temporal}")
        if self.spatial:
            parts.append(f"Location: {self.spatial}")
        if len(parts) > 1:
            return f"{parts[0]} ({'; '.join(parts[1:])})"
        return self.content


@dataclass
class Topic:
    """
    Topic data structure.

    A topic is an abstraction of multiple related episodes, representing
    a recurring pattern, common context, or persistent activity.
    """
    topic_id: str
    title: str
    summary: str
    episode_ids: List[str]
    timestamp: datetime.datetime
    user_id_list: List[str]

    participants: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

    def __post_init__(self):
        if not self.topic_id:
            raise ValueError("topic_id is required")
        if not self.title:
            raise ValueError("title is required")
        if not self.summary:
            raise ValueError("summary is required")
        if not self.episode_ids:
            raise ValueError("episode_ids is required")

    def __repr__(self) -> str:
        return f"Topic(topic_id={self.topic_id}, title={self.title}, episode_count={len(self.episode_ids)})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "summary": self.summary,
            "episode_ids": self.episode_ids,
            "timestamp": to_iso_format(self.timestamp),
            "user_id_list": self.user_id_list,
            "participants": self.participants,
            "keywords": self.keywords,
        }
