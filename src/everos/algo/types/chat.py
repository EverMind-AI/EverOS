"""Chat conversation wire types — ``ChatMessage`` is the common currency across chat and agent pipelines."""

from __future__ import annotations

__all__ = ["ChatMessage"]

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Single chat-style conversation message.

    ``kind="text"`` is the discriminator for ``ConversationItem``. ``content`` is text: a host turns images, audio and
    files into text before handing a message over.
    """

    kind: Literal["text"] = "text"
    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: int  # Unix epoch milliseconds
    sender_id: str
    sender_name: str | None = Field(
        default=None,
        description="Human-readable speaker label rendered into LLM prompts",
    )

    model_config = ConfigDict(extra="ignore")
