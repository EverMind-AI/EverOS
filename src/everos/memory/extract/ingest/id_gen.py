"""Deterministic, human-readable ``message_id`` generation.

Format: ``m_<session_id>_<timestamp_ms>_<digest>``, where ``digest`` is the
first 12 hex digits of the SHA-256 of the sender's id, a line break and the
content as sent (:func:`content_text`) -- and, for the second and later copies
of the same message (same sender, time and content) inside one batch, a line
break and the copy's number (1, 2, ...).

Human-readable so logs / debugging / md entries stay greppable. Unique per
distinct message: two messages that differ in sender or content never share
an id, even when their timestamps are equal and each came alone in its own
batch (the old ``<index in its batch>`` suffix gave both ``000``, and the
second was dropped as the first's re-send). Deterministic and independent of
the message's place in its batch, so a re-send of the same message -- in the
same batch or another -- gets the id it had and is taken once.
"""

from __future__ import annotations

__all__ = ["content_text", "gen_message_id"]

import hashlib
import json
from typing import Any

_DIGEST_HEX = 12
"""48 bits: unique among the messages one session holds at one timestamp."""


def gen_message_id(
    session_id: str, timestamp_ms: int, sender_id: str, content: Any, copy: int = 0
) -> str:
    """Return ``m_<session_id>_<timestamp_ms>_<digest>`` (see the module doc).

    Args:
        session_id: The message's session.
        timestamp_ms: Its time, epoch milliseconds.
        sender_id: Who sent it.
        content: Its content as sent: a string, or the content items.
        copy: ``0`` for the first copy of this message in its batch, ``n``
            for the ``n``-th identical one after it.
    """
    text = f"{sender_id}\n{content_text(content)}"
    if copy:
        text = f"{text}\n{copy}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"m_{session_id}_{timestamp_ms}_{digest}"


def content_text(content: Any) -> str:
    """A message's content as one string: the text itself, or the items as JSON.

    Content items are canonical JSON (sorted keys, no spaces, unset fields
    left out), so the same items always give the same string.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    items: list[Any] = []
    for item in content:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        items.append({k: v for k, v in data.items() if v is not None})
    return json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
