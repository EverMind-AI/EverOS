"""``memorize_inbox`` — durable queue for ``async_mode`` memorize requests.

One row per accepted ``/add`` or ``/flush`` request. The route inserts the
row (``queued``) before returning 202; the in-process worker claims rows
in ``id`` order, never two of one session at a time, calls ``memorize()``
and lands each row in ``success`` or ``failed``. Within a session ``id``
is the delivery sequence, which is what keeps its ``/add`` batches and its
``/flush`` in the order the client sent them; sessions run side by side.

Rows are retained for ``memorize.async_queue.row_retention_hours`` so
``GET /tasks/{task_id}`` can report the outcome, then purged whatever
their status (``processing`` excepted).
"""

from __future__ import annotations

from sqlalchemy import Index

from everos.core.persistence.sqlite import BaseTable, Field


class MemorizeInbox(BaseTable, table=True):
    """One accepted async memorize request; ``id`` orders a session's rows."""

    __tablename__ = "memorize_inbox"  # type: ignore[assignment]
    __table_args__ = (
        # Per-session cap check + session summary.
        Index(
            "ix_memorize_inbox_scope_status",
            "app_id",
            "project_id",
            "session_id",
            "status",
        ),
        # Worker claim: smallest id among queued rows.
        Index("ix_memorize_inbox_status_id", "status", "id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    request_id: str = Field(unique=True)
    """The HTTP request id of the accepting call; the status lookup key."""
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    session_id: str
    is_final: bool = False
    """``True`` for ``/flush`` rows (``messages`` is empty)."""
    defer_extraction: bool = False
    payload_json: str
    """The request body as ``memorize()`` expects it, serialised verbatim.
    Emptied once the row is delivered; a ``failed`` row keeps it."""
    message_count: int = 0
    status: str = Field(default="queued")
    """``queued`` | ``processing`` | ``success`` | ``failed``."""
    attempt: int = 0
    last_error: str | None = None
