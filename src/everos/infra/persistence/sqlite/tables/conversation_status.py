"""``conversation_status`` — window pointer per (app, project, session, track).

The window pointer is scoped by ``app_id`` / ``project_id`` so the same
``session_id`` may recur in different spaces without colliding; those two
segments lead the composite ``UniqueConstraint``.
"""

from __future__ import annotations

from sqlalchemy import UniqueConstraint

from everos.component.utils.datetime import UtcDatetime
from everos.core.persistence.sqlite import BaseTable, Field
from everos.core.persistence.sqlite.base import UtcDateTimeColumn


class ConversationStatus(BaseTable, table=True):
    """One row per (app, project, session, track). Tracks latest msg / memcell ts."""

    __tablename__ = "conversation_status"  # type: ignore[assignment]
    __table_args__ = (
        UniqueConstraint(
            "app_id",
            "project_id",
            "session_id",
            "track",
            name="uq_conversation_status_session_track",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    """App / project scope segments (default ``"default"``)."""
    session_id: str = Field(index=True)
    track: str
    last_message_ts: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    last_memcell_ts: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
