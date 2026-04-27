from enum import Enum
from dataclasses import dataclass
from typing import Dict


class MessageRole(Enum):
    """Message role enumeration"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """Chat message data class"""

    role: MessageRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format"""
        return {"role": self.role.value, "content": self.content}
