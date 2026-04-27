"""
Group Memorize Event

Published when a group add request is accepted.
Enterprise listeners can use this to record sender↔group associations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from core.events import BaseEvent


@dataclass
class GroupMemorizeEvent(BaseEvent):
    """
    Event emitted after a group memorize request is accepted.

    Attributes:
        group_id: Group identifier
        sender_ids: List of unique sender IDs from the messages
    """

    group_id: str = ""
    sender_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls: Type["GroupMemorizeEvent"], data: Dict[str, Any]
    ) -> "GroupMemorizeEvent":
        return cls(
            event_id=data.get("event_id", ""),
            created_at=data.get("created_at", ""),
            group_id=data.get("group_id", ""),
            sender_ids=data.get("sender_ids", []),
        )
