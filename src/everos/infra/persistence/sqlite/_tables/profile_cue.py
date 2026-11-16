"""``profile_cue`` -- what each segment's episode call marked for a person's profile.

One row per ``(memcell_id, person_id)``: the person's own messages and the
pictures they posted that the call marked as saying something lasting about
them, as JSON arrays of message ids and content hashes. Every human speaker of
a segment extracted while a profile of its space read cues has a row, an
empty one when nothing was marked; a segment with no row was never marked.
``read_at`` is when the person's profile read the segment.

Derived state only: markdown records none of it, and a store rebuilt from
markdown has none (its profiles then read their segments in full). Rows are
never deleted by a forget -- a forgotten message or picture is skipped by
every read, as in the memcell payload -- only with their session's purge.
"""

from __future__ import annotations

from sqlmodel import Field

from everos.core.utils.datetime import UtcDatetime
from everos.infra.persistence.sqlite._base.base import BaseTable, UtcDateTimeColumn


class ProfileCue(BaseTable, table=True):
    """One segment's cues for one person. PK ``(memcell_id, person_id)``."""

    __tablename__ = "profile_cue"  # type: ignore[assignment]

    memcell_id: str = Field(primary_key=True)
    person_id: str = Field(primary_key=True)
    """The speaker's ``sender_id``: the user, or the member in a group."""
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    session_id: str = Field(index=True)
    message_ids_json: str = "[]"
    """The person's marked messages, by id, in conversation order."""
    image_ids_json: str = "[]"
    """The pictures they posted that were marked, by content hash."""
    read_at: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    """When the person's profile read the segment; ``None`` until it has."""
