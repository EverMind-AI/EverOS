"""``memcell`` — metadata + payload archive for boundary-detected MemCells.

Holds ``message_ids_json`` / ``sender_ids_json`` (JSON arrays of audit
ids) plus ``payload_json`` — the full :class:`everalgo.types.MemCell`
serialised via ``model_dump_json``. The payload is what
``unprocessed_buffer`` cannot keep (boundary's delete-then-insert clears
the staging slice once messages fold into a cell): downstream offline
strategies that need the raw chat messages (e.g. profile extraction)
deserialise the payload back into an algo ``MemCell``. Episode markdown
still carries the LLM-synthesised narrative; ``payload_json`` is the
chat-stream archive that narrative was distilled from.

``extraction_retry`` marks a segment whose extraction failed: the session's
next add or flush extracts it again (``SqliteMemcellStore``).

A ``system.db`` written by an earlier 2.0 build also has three nullable
``case_*`` columns (the agent-case extraction stamp); nothing reads or
writes them now.
"""

from __future__ import annotations

from sqlalchemy import Index
from sqlmodel import Field

from everos.core.utils.datetime import UtcDatetime
from everos.infra.persistence.sqlite._base.base import BaseTable, UtcDateTimeColumn


class Memcell(BaseTable, table=True):
    """One row per MemCell. PK ``memcell_id`` (uuid4)."""

    __tablename__ = "memcell"  # type: ignore[assignment]
    __table_args__ = (
        # Scope-first composite: app/project partition the lookup before the
        # session window so cross-(app, project) rows never share an index slot.
        Index(
            "ix_memcell_session",
            "app_id",
            "project_id",
            "session_id",
            "track",
            "timestamp",
        ),
    )

    memcell_id: str = Field(primary_key=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    """App / project scope segments. Default to ``"default"`` so the column is
    always populated; callers in a non-default space pass real ids."""
    session_id: str = Field(index=True)
    track: str
    raw_type: str
    message_ids_json: str
    sender_ids_json: str
    payload_json: str
    """``MemCell.model_dump_json()`` — the full algo-side MemCell (items =
    chat messages) serialised at boundary time so offline
    strategies can deserialise it back into an algo MemCell long after
    ``unprocessed_buffer`` has dropped the staging rows."""
    timestamp: UtcDatetime = Field(sa_type=UtcDateTimeColumn)
    extraction_retry: str | None = None
    """JSON of a segment whose extraction failed and is to be tried again
    (failures so far, the group it was cut in, its pictures' client names);
    ``NULL`` for every other segment, a legacy one included. Added after the
    table's first release (``schema._ADDED_COLUMNS``)."""
