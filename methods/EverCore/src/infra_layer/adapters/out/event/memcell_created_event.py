# -*- coding: utf-8 -*-
"""
MemCell Creation Event Class

Used to report MemCell creation information, inherits from BaseEvent, supports JSON and BSON serialization/deserialization.
This is the basic event for the open-source version, containing only core fields. The enterprise version can extend this class to include additional fields.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from core.events import BaseEvent


@dataclass
class MemCellCreatedEvent(BaseEvent):
    """
    MemCell Creation Event (open-source version)

    Used to record basic information about MemCell creation.
    Inherits from BaseEvent, automatically gaining event_id and created_at fields.

    Attributes:
        memcell_id: MemCell ID
        timestamp: Timestamp when the event occurred (optional, Unix timestamp in milliseconds)
        extend: Extension field for storing additional information (optional)
    """

    # Business fields
    memcell_id: str = ""
    timestamp: Optional[int] = None
    extend: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls: Type['MemCellCreatedEvent'], data: Dict[str, Any]
    ) -> 'MemCellCreatedEvent':
        """
        Create an instance from a dictionary

        Args:
            data: Dictionary containing event data

        Returns:
            MemCellCreatedEvent: Instance of the class

        Raises:
            KeyError: Missing required fields
            TypeError: Incorrect field types
        """
        return cls(
            # Base class fields
            event_id=data.get("event_id", ""),
            created_at=data.get("created_at", ""),
            # Business fields
            memcell_id=data.get("memcell_id", ""),
            timestamp=data.get("timestamp"),
            extend=data.get("extend", {}),
        )

    def __repr__(self) -> str:
        """Return string representation of the object"""
        return (
            f"MemCellCreatedEvent("
            f"event_id={self.event_id!r}, "
            f"memcell_id={self.memcell_id!r}, "
            f"timestamp={self.timestamp}"
            f")"
        )
