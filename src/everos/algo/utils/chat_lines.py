"""Chat messages as the transcript lines every operator shows a model, one shape for all of them."""

from __future__ import annotations

__all__ = ["render_chat_line", "speaker"]

from typing import Protocol

from everos.algo.utils.errors import InvalidInputError
from everos.algo.utils.timestamps import format_utc_time


class _ChatLineSource(Protocol):
    """The fields of ``everos.algo.types.ChatMessage`` a line reads; ``utils`` imports nothing from ``types``."""

    @property
    def id(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def timestamp(self) -> int: ...

    @property
    def sender_id(self) -> str: ...

    @property
    def sender_name(self) -> str | None: ...


def speaker(message: _ChatLineSource) -> str:
    """Return the label a line names its author by: ``sender_name``, else ``sender_id``.

    Falling back to ``role`` instead would merge the distinct users of a group chat into one ``user``.
    """
    return message.sender_name or message.sender_id


def render_chat_line(message: _ChatLineSource) -> str:
    """Render one message as ``[2024-04-10 14:00 UTC (Wednesday)] Alice: text``.

    The time carries its weekday because the model resolves relative times ("last Friday") against each message's
    own timestamp, and without the weekday it miscounts weeks. ``content`` is placed verbatim, newlines included.
    Skipping a message with empty ``content`` is the caller's call: some operators drop it, others keep its index.

    Args:
        message: A ``ChatMessage``, or any object with its fields.

    Returns:
        The line, with no trailing newline.

    Raises:
        InvalidInputError: If ``message.timestamp`` is 0, an unset time that would render as 1970-01-01. The message
            names the chat message's id, never its text or speaker.
    """
    if not message.timestamp:
        raise InvalidInputError(
            f"chat message {message.id!r} has no timestamp: every ChatMessage must carry a positive "
            "epoch-millisecond timestamp; validate or fill it at ingress before calling the operator"
        )
    return f"[{format_utc_time(message.timestamp, weekday=True)}] {speaker(message)}: {message.content}"
