"""Chat messages as the text the boundary prompts and the hard limits see.

The batch prompt numbers its records, escapes XML delimiters, and indents continuation lines so message text
cannot introduce a top-level record. Empty messages retain their indices as placeholders. The step prompt keeps
the original unnumbered lines and drops empty messages. Token counting reads the raw ``speaker: text`` view.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from everos.algo.utils import count_tokens, render_chat_line, speaker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.types import ChatMessage


def _format_messages_with_indices(messages: Sequence[ChatMessage]) -> str:
    """Render escaped records with 1-based indices, retaining empty messages without reading their timestamps.

    Raises:
        InvalidInputError: If a message with text has no timestamp (0).
    """
    return "\n".join(
        f"[{index}] {_batch_record(message)}"
        for index, message in enumerate(messages, start=1)
    )


def _batch_record(message: ChatMessage) -> str:
    """Keep message text inside its XML block and continuation lines inside their numbered record.

    Raises:
        InvalidInputError: If a message with text has no timestamp (0).
    """
    if not message.content:
        return "[empty message]"
    return "\n  ".join(escape(render_chat_line(message), quote=False).splitlines())


def _format_messages_step(messages: Sequence[ChatMessage]) -> str:
    """Render one line per message with text, unnumbered: the step prompt answers a single ``should_end``.

    Raises:
        InvalidInputError: If a message with text has no timestamp (0).
    """
    return "\n".join(
        render_chat_line(message) for message in messages if message.content
    )


def _count_history_tokens(messages: Sequence[ChatMessage]) -> int:
    """Sum of each message's ``speaker: text`` token count (per message, not over the joined text)."""
    total = 0
    for message in messages:
        label = speaker(message)
        total += count_tokens(
            f"{label}: {message.content}" if label else message.content
        )
    return total
