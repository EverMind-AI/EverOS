"""``sender_name`` — the latest display name per (app, project, sender).

One row per sender in a space; a newer non-blank name overwrites the row, so
it always holds the name the sender went by last. Ingest reads it to name a
message that arrived without ``sender_name``.
"""

from __future__ import annotations

from sqlmodel import Field

from everos.infra.persistence.sqlite._base.base import BaseTable


class SenderName(BaseTable, table=True):
    """The name ``sender_id`` last went by in ``(app_id, project_id)``."""

    __tablename__ = "sender_name"  # type: ignore[assignment]

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    sender_id: str = Field(primary_key=True)
    sender_name: str
