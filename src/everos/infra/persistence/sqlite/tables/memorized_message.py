"""``memorized_message`` — timestamps of messages already carved into a memcell.

Local patch. Clients may re-post a session's whole visible transcript on every
turn (the OpenClaw plugin does, on ``agent_end``). Once a message is in a
memcell it has left ``unprocessed_buffer``, so the buffer's ``message_id`` PK
can no longer dedupe it, and ``message_id`` itself is not stable across
re-posts (its ``_idx`` is the position inside the request). The message
timestamp is stable, so ingest drops any message whose
``(app, project, session, timestamp_ms)`` is recorded here.
"""

from __future__ import annotations

from sqlmodel import Field, SQLModel


class MemorizedMessage(SQLModel, table=True):
    """One row per message timestamp already inside a memcell of a session."""

    __tablename__ = "memorized_message"  # type: ignore[assignment]

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    timestamp_ms: int = Field(primary_key=True)
