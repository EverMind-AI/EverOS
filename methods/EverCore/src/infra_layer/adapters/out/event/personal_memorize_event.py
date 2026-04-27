"""
Personal Memorize Event

Published when a personal add request is accepted.
Enterprise listeners can use this to record user↔session associations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Type

from core.events import BaseEvent


@dataclass
class PersonalMemorizeEvent(BaseEvent):
    """
    Event emitted after a personal memorize request is accepted.

    Attributes:
        user_id: Owner user ID
        session_id: Session identifier (may be DEFAULT_SESSION_ID if not provided)
        group_id: Auto-generated group ID for this user
    """

    user_id: str = ""
    session_id: str = ""
    group_id: str = ""

    @classmethod
    def from_dict(
        cls: Type["PersonalMemorizeEvent"], data: Dict[str, Any]
    ) -> "PersonalMemorizeEvent":
        return cls(
            event_id=data.get("event_id", ""),
            created_at=data.get("created_at", ""),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            group_id=data.get("group_id", ""),
        )
