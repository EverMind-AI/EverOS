"""Input helpers shared by the user-memory operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.types.chat import ChatMessage

if TYPE_CHECKING:
    from everos.algo.types.conversation import MemCell

__all__ = ["chat_messages"]


def chat_messages(memcell: MemCell) -> list[ChatMessage]:
    """Filter ``memcell.items`` to ``ChatMessage`` entries only.

    User-memory extractors silently skip ``ToolCallRequest`` / ``ToolCallResult`` — this is an
    explicit contract supporting the agent → user-memory pipeline. The
    caller need not pre-filter; an AgentMemCell-shaped ``MemCell`` is acceptable input.
    """
    return [item for item in memcell.items if isinstance(item, ChatMessage)]
