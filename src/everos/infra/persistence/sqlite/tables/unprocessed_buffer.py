"""``unprocessed_buffer`` — chat-stream messages waiting on boundary detection.

Schema property: presence in the table = pending; absence = consumed.
There is no ``consumed`` column. Pipeline uses ``replace(session, track,
remaining)`` to atomically rewrite the (session, track) slice each turn.
"""

from __future__ import annotations

from sqlalchemy import Index

from everos.component.utils.datetime import UtcDatetime
from everos.core.persistence.sqlite import BaseTable, Field
from everos.core.persistence.sqlite.base import UtcDateTimeColumn


class UnprocessedBuffer(BaseTable, table=True):
    """One row per unprocessed message. PK ``message_id``."""

    __tablename__ = "unprocessed_buffer"  # type: ignore[assignment]
    __table_args__ = (
        # Scope-first composite: app/project partition the (session, track)
        # staging slice so different spaces never share a buffer window.
        Index(
            "ix_unprocessed_buffer_lookup",
            "app_id",
            "project_id",
            "session_id",
            "track",
            "timestamp",
        ),
    )

    message_id: str = Field(primary_key=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    """App / project scope segments (default ``"default"``)."""
    session_id: str = Field(index=True)
    track: str = Field(index=True)
    sender_id: str
    sender_name: str | None = None
    role: str
    timestamp: UtcDatetime = Field(sa_type=UtcDateTimeColumn)
    # JSON-serialised raw ContentItem list (mirrors src_old
    # RawMessage.content_items). Keeps the original multimodal payload
    # available so a future parser can reach back to image / audio / etc.
    content_items_json: str
    # Derived plain-text concatenation of ``type=text`` entries — what
    # downstream LLM-facing extractors and md writer consume today.
    text: str
    tool_calls_json: str | None = None
    tool_call_id: str | None = None
