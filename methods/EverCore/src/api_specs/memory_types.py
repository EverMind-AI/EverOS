"""
Memory types module

This module contains the definitions of memory types and related data structures, just for extraction.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from common_utils.datetime_utils import to_iso_format


class ScenarioType(str, Enum):
    """Conversation scenario types for memory extraction."""

    SOLO = "solo"  # 1 user + N agents scenario
    TEAM = "team"  # Multi-user + agents scenario


# Import after ScenarioType definition to avoid circular import
# (memory_models imports ScenarioType from this module)
from api_specs.memory_models import MemoryType  # noqa: E402


class RawDataType(Enum):
    """Types of content that can be processed."""

    CONVERSATION = "Conversation"
    AGENTCONVERSATION = "AgentConversation"

    @classmethod
    def from_string(cls, type_str: Optional[str]) -> Optional['RawDataType']:
        """
        Convert string type to RawDataType enum

        Args:
            type_str: Type string, such as "Conversation", "Email", etc.

        Returns:
            RawDataType enum value, returns None if conversion fails
        """
        if not type_str:
            return None

        try:
            # Convert string to enum name format (e.g., "Conversation" -> "CONVERSATION")
            enum_name = type_str.upper()
            return getattr(cls, enum_name)

        except AttributeError:
            # If no matching enum is found, return None
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.error(f"No matching RawDataType found: {type_str}, returning None")
            return None
        except Exception as e:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to convert type field: {type_str}, error: {e}")
            return None


class ParentType(str, Enum):
    """Parent memory type for Foresight/AtomicFact."""

    MEMCELL = "memcell"
    EPISODE = "episode"


def get_text_from_content_items(content_items: Any) -> str:
    """Extract text from a content items list.

    Takes a content[] list (v1 API format) and returns the concatenated text.
    For type="text" items, uses the text field directly.
    For non-text types, reads parsed_summary from item itself (embedded by enrich provider),
    then formats as [TYPE: name | Summary: ...] to preserve file metadata for LLM consumption.

    Args:
        content_items: List of content item dicts [{type: "text", text: "..."}],
                       or a plain string (legacy fallback).
                       Non-text items may contain parsed_summary/parsed_content fields
                       embedded by the enrich provider.

    Returns:
        str: Extracted text content, space-joined across items
    """
    if isinstance(content_items, str):
        return content_items
    if not isinstance(content_items, list):
        return ""
    texts = []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text") or item.get("content", "")
            if text:
                texts.append(text)
        else:
            item_type = (item.get("type") or "file").upper()
            name = item.get("name", "")
            parsed_summary = item.get("parsed_summary")
            if name and parsed_summary:
                texts.append(f"[{item_type}: {name} | Summary: {parsed_summary}]")
            elif name:
                texts.append(f"[{item_type}: {name}]")
            elif parsed_summary:
                texts.append(f"[{item_type} | Summary: {parsed_summary}]")
            else:
                texts.append(f"[{item_type}]")
    return " ".join(texts) if texts else ""


def is_intermediate_agent_step(msg: Dict[str, Any]) -> bool:
    """Check if a message is an intermediate agent step (tool call or tool response).

    Intermediate steps are:
    - role="tool": Tool execution results
    - role="assistant" WITH tool_calls: Intermediate tool invocations
    """
    role = msg.get("role", "")
    if role == "tool":
        return True
    if role == "assistant" and msg.get("tool_calls"):
        return True
    return False


@dataclass
class MemCell:
    # TODO: Name conflict - should add BO suffix (such as MemCellBO) to distinguish between business objects and document objects
    """
    Boundary detection result following the specified schema.

    This class represents the result of boundary detection analysis
    and contains all the required fields for memory storage.
    """

    # Required fields (must come before fields with default values)
    user_id_list: List[str]
    # Each item is { "message": {API message with content[]} }
    # Access message via item["message"], extract text via get_text_from_content_items(msg["content"])
    original_data: List[Dict[str, Any]]
    timestamp: datetime

    # Optional fields
    event_id: Optional[str] = None  # Generated by database when saving
    group_id: Optional[str] = None
    # NOTE: participants and sender_ids currently hold the same values (both are sender_id).
    # participants is not yet implemented as display names; it is populated with sender_ids
    # as a placeholder. Once display-name resolution is available, participants will carry
    # human-readable names while sender_ids will remain the raw identifiers.
    participants: Optional[List[str]] = None
    sender_ids: Optional[List[str]] = None
    type: Optional[RawDataType] = None

    # Cached filtered data (excluded from __init__)
    _conversation_data_cache: Optional[List[Dict[str, Any]]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """Validate the result after initialization."""
        if not self.original_data:
            raise ValueError("original_data is required")

    @property
    def conversation_data(self) -> List[Dict[str, Any]]:
        """Return conversation data with tool calls/responses filtered out for agent conversations.

        For AGENTCONVERSATION type, intermediate agent steps (tool calls and tool responses)
        are excluded. For other types, returns original_data as-is.
        Use original_data directly when full trajectory is needed (e.g. AgentCase extraction).
        """
        if self._conversation_data_cache is not None:
            return self._conversation_data_cache

        if self.type != RawDataType.AGENTCONVERSATION:
            self._conversation_data_cache = self.original_data
        else:
            self._conversation_data_cache = [
                item
                for item in self.original_data
                if not is_intermediate_agent_step(
                    item.get("message", item) if isinstance(item, dict) else item
                )
            ]
        return self._conversation_data_cache

    def __repr__(self) -> str:
        return f"MemCell(event_id={self.event_id!r}, original_data={self.original_data!r}, timestamp={self.timestamp!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id if self.event_id else None,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": to_iso_format(self.timestamp),
            "group_id": self.group_id,
            "participants": self.participants,
            "sender_ids": self.sender_ids,
            "type": str(self.type.value) if self.type else None,
        }


@dataclass
class BaseMemory:
    """
    Base class for all memory types.
    Contains common fields shared by all memory types.
    """

    memory_type: Union[MemoryType, str]
    user_id: str
    timestamp: datetime

    ori_event_id_list: Optional[List[str]] = None
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    sender_ids: Optional[List[str]] = None
    type: Optional[RawDataType] = None
    keywords: Optional[List[str]] = None
    linked_entities: Optional[List[str]] = None
    user_name: Optional[str] = None
    extend: Optional[Dict[str, Any]] = None

    # vector and model
    vector_model: Optional[str] = None
    vector: Optional[List[float]] = None

    # ID field for retrieval
    id: Optional[str] = None

    # Retrieval-related fields
    score: Optional[float] = None
    original_data: Optional[List[Dict[str, Any]]] = None

    def _format_timestamp(self) -> Optional[str]:
        """Format timestamp to ISO string"""
        if not self.timestamp:
            return None
        if isinstance(self.timestamp, str):
            return self.timestamp if self.timestamp else None
        try:
            return to_iso_format(self.timestamp)
        except Exception:
            return str(self.timestamp) if self.timestamp else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value if self.memory_type else None,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "timestamp": self._format_timestamp(),
            "group_id": self.group_id,
            "participants": self.participants,
            "sender_ids": self.sender_ids,
            "type": self.type.value if self.type else None,
            "keywords": self.keywords,
            "linked_entities": self.linked_entities,
            "score": self.score,
            "original_data": self.original_data,
            "extend": self.extend,
        }


@dataclass
class EpisodeMemory(BaseMemory):
    """Episode memory - narrative memory of events."""

    id: Optional[str] = None
    subject: Optional[str] = None
    summary: Optional[str] = None
    episode: Optional[str] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["id"] = self.id
        d["subject"] = self.subject
        d["summary"] = self.summary
        d["episode"] = self.episode
        d["parent_type"] = self.parent_type
        d["parent_id"] = self.parent_id
        return d


@dataclass
class AtomicFact(BaseMemory):
    """Atomic fact - atomic facts extracted from MemCell/conversation."""

    time: Optional[str] = None
    atomic_fact: Optional[Union[str, List[str]]] = None
    fact_embeddings: Optional[List[List[float]]] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.time:
            d["time"] = self.time
        if self.atomic_fact:
            d["atomic_fact"] = self.atomic_fact
        if self.fact_embeddings:
            d["fact_embeddings"] = self.fact_embeddings
        d["parent_type"] = self.parent_type
        if self.parent_id:
            d["parent_id"] = self.parent_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicFact":
        """Create from dictionary."""
        return cls(
            memory_type=MemoryType.from_string(data.get("memory_type")),
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp"),
            time=data.get("time", ""),
            atomic_fact=data.get("atomic_fact", []),
            fact_embeddings=data.get("fact_embeddings"),
            parent_type=data.get("parent_type"),
            parent_id=data.get("parent_id"),
        )


@dataclass
class Foresight(BaseMemory):
    """Foresight prediction memory extracted from MemCell/conversation."""

    foresight: Optional[str] = None
    evidence: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_days: Optional[int] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["foresight"] = self.foresight
        d["evidence"] = self.evidence
        d["start_time"] = self.start_time
        d["end_time"] = self.end_time
        d["duration_days"] = self.duration_days
        d["parent_type"] = self.parent_type
        d["parent_id"] = self.parent_id
        return d


@dataclass
class AgentCase(BaseMemory):
    """Agent experience extracted from an agent conversation MemCell.

    Each MemCell produces at most one experience. Multiple conversation turns
    that solve the same problem are synthesized into a single experience record.

    Used both as extraction output (with vector/vector_model) and as retrieval
    result (with score/keywords from BaseMemory).

    Fields:
    - task_intent: Retrieval key - the task rewritten as a standalone statement
    - approach: Natural-language numbered steps with inline decisions, results, and lessons
    - quality_score: How well the agent completed this task (0.0-1.0)
    """

    task_intent: Optional[str] = None
    approach: Optional[str] = None
    quality_score: Optional[float] = None  # 0.0-1.0, task completion quality
    key_insight: Optional[str] = None  # pivotal strategy shift or decision
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "task_intent": self.task_intent,
                "approach": self.approach,
                "quality_score": self.quality_score,
                "key_insight": self.key_insight,
                "parent_type": self.parent_type,
                "parent_id": self.parent_id,
            }
        )
        return d


@dataclass
class AgentSkill(BaseMemory):
    """Reusable skill derived from clustered AgentCases.

    Skills belong to a specific cluster and user (agent owner).
    user_id and timestamp are Optional because they default from the extraction context.
    """

    # Override BaseMemory required fields
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    confidence: float = 0.0
    cluster_id: Optional[str] = None
    maturity_score: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "name": self.name,
                "description": self.description,
                "content": self.content,
                "confidence": self.confidence,
                "cluster_id": self.cluster_id,
                "maturity_score": self.maturity_score,
            }
        )
        return d


@dataclass
class ProfileMemory(BaseMemory):
    """Profile Memory - Explicit information + Implicit traits.

    explicit_info item: {"category": str, "description": str, "evidence": str, "sources": [str]}
    implicit_traits item: {"trait": str, "description": str, "basis": str, "evidence": str, "sources": [str]}
    """

    explicit_info: List[Dict[str, Any]] = field(default_factory=list)
    implicit_traits: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    processed_episode_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.memory_type = MemoryType.PROFILE
        if self.last_updated is None:
            self.last_updated = datetime.now().astimezone()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value if self.memory_type else None,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "group_id": self.group_id,
            "explicit_info": list(self.explicit_info),
            "implicit_traits": list(self.implicit_traits),
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "processed_episode_ids": self.processed_episode_ids,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], user_id: str = "", group_id: str = ""
    ) -> "ProfileMemory":
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)

        return cls(
            memory_type=MemoryType.PROFILE,
            user_id=user_id or data.get("user_id", ""),
            group_id=group_id or data.get("group_id", ""),
            timestamp=datetime.now().astimezone(),
            ori_event_id_list=data.get("ori_event_id_list", []),
            explicit_info=data.get("explicit_info", []),
            implicit_traits=data.get("implicit_traits", []),
            last_updated=last_updated,
            processed_episode_ids=data.get("processed_episode_ids", []),
        )

    def total_items(self) -> int:
        return len(self.explicit_info) + len(self.implicit_traits)

    def get_all_source_ids(self) -> set:
        ids = set()
        for item in self.explicit_info + self.implicit_traits:
            for s in item.get("sources", []):
                s = str(s)
                if "|" in s:
                    s = s.rsplit("|", 1)[-1].strip()
                if s:
                    ids.add(s)
        return ids

    def to_readable_document(self) -> str:
        lines = [
            "=" * 50,
            "User Profile Document",
            f"Last Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M') if self.last_updated else 'N/A'}",
            f"Total {self.total_items()} items (Explicit: {len(self.explicit_info)}, Implicit: {len(self.implicit_traits)})",
            "=" * 50,
        ]

        if self.explicit_info:
            lines.append("\n[Explicit Info]")
            categories: Dict[str, list] = {}
            for info in self.explicit_info:
                categories.setdefault(info.get("category", ""), []).append(info)
            for cat, infos in categories.items():
                lines.append(f"  [{cat}]")
                for info in infos:
                    desc = info.get("description", "")
                    evidence = info.get("evidence", "")
                    if evidence:
                        lines.append(f"    - {desc} (evidence: {evidence})")
                    else:
                        lines.append(f"    - {desc}")

        if self.implicit_traits:
            lines.append("\n[Implicit Traits]")
            for i, trait in enumerate(self.implicit_traits, 1):
                lines.append(f"  {i}. {trait.get('trait', '')}")
                lines.append(f"     {trait.get('description', '')}")
                lines.append(f"     - basis: {trait.get('basis', '')}")
                evidence = trait.get("evidence", "")
                if evidence:
                    lines.append(f"     - evidence: {evidence}")

        return "\n".join(lines)

    def to_readable_profile(self) -> str:
        lines = []

        if self.explicit_info:
            lines.append("[Explicit Info]")
            categories: Dict[str, list] = {}
            for info in self.explicit_info:
                categories.setdefault(info.get("category", ""), []).append(info)
            for cat, infos in categories.items():
                lines.append(f"  {cat}:")
                for info in infos:
                    lines.append(f"    - {info.get('description', '')}")

        if self.implicit_traits:
            if lines:
                lines.append("")
            lines.append("[Implicit Traits]")
            for trait in self.implicit_traits:
                lines.append(f"  - {trait.get('trait', '')}: {trait.get('description', '')}")

        return "\n".join(lines) if lines else "No profile data yet."


# Union type for search/retrieve API response
RetrieveMemoryModel = Union[EpisodeMemory, AtomicFact, Foresight, AgentSkill, AgentCase]
