"""Conversation wire types shared by chat and agent pipelines — ``MemCell`` and ``ConversationItem``."""

from __future__ import annotations

__all__ = ["ConversationItem", "MemCell"]

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from everos.algo.types.agent import ToolCallRequest, ToolCallResult
from everos.algo.types.chat import ChatMessage

ConversationItem = Annotated[
    ChatMessage | ToolCallRequest | ToolCallResult,
    Field(discriminator="kind"),
]
"""One ordered step in a conversation or agent trajectory; discriminated by ``kind``."""


class MemCell(BaseModel):
    """A coherent slice produced by boundary detection.

    ``items`` is ordered; chat scenarios contain only ``ChatMessage`` while agent scenarios may also carry
    ``ToolCallRequest`` / ``ToolCallResult``. User-memory extractors silently skip non-``ChatMessage`` items.
    ``timestamp`` is Unix epoch ms of the closing item in the slice.
    """

    items: list[ConversationItem]
    timestamp: int  # Unix epoch ms — closing time of the slice

    model_config = ConfigDict(extra="ignore")
