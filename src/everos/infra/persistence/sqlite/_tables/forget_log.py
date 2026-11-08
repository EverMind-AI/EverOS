"""``forget_log`` — one row per forget, the record an undo reads.

The design keeps every memory on disk and makes forgetting a *mark*, so an
undo is a status change. That works for the kinds whose markdown can carry the
mark itself (an episode's ``deprecated_entries``, a picture memory's
``deprecated_by``). It does not work for the two things a forget changes
outside a daily-log entry:

- the picture file's own status (``image_original.forgotten_at``, which an
  undo clears directly), and
- the **profile items** a forget removed from ``user.md`` — everalgo's
  document is replaced wholesale on every synthesis and has no per-item mark to
  append to, so nothing on disk remembers what was there.

This table is that memory. One row per forget, carrying what it touched and
enough of the removed material to put it back. Rows are append-only: an undo
writes ``undone_at`` and a new row would be a new forget, so replaying the log
in order tells the whole story of what was hidden and shown again.

The row is scoped like every other memory (``app_id`` / ``project_id`` /
``owner_id``) and keyed by what the forget addressed — an ``image`` hash or a
``session`` id — so an undo finds exactly the forget it is undoing.
"""

from __future__ import annotations

from sqlalchemy import Index
from sqlmodel import Field

from everos.core.utils.datetime import UtcDatetime
from everos.infra.persistence.sqlite._base.base import BaseTable, UtcDateTimeColumn


class ForgetLog(BaseTable, table=True):
    """One forget. PK ``forget_id``; append-only, undone by a column not a delete."""

    __tablename__ = "forget_log"  # type: ignore[assignment]
    __table_args__ = (
        # An undo looks up the owner's forget of one target, newest first.
        Index(
            "ix_forget_log_target",
            "app_id",
            "project_id",
            "owner_id",
            "target_kind",
            "target_id",
            "created_at",
        ),
    )

    forget_id: str = Field(primary_key=True)
    """``fg_<hex>`` — minted here, returned to the caller so an undo can name it."""

    target_kind: str
    """``image`` (a picture, by content hash) or ``session`` (a conversation)."""

    target_id: str
    """The content hash, or the session id."""

    app_id: str = Field(index=True)
    project_id: str = Field(index=True)
    owner_id: str = Field(index=True)
    """Whose memories were forgotten. A forget is always one owner's: the
    picture file may be shared, the memories about it are not."""

    owner_type: str = "user"

    reason: str | None = None
    """The caller's free-text reason. Logged, never used as a mark."""

    entry_ids_json: str = "[]"
    """Picture-memory entry ids the forget marked, for the undo's report."""

    episode_ids_json: str = "[]"
    """Episode entry ids the forget re-wrote, and their superseded originals:
    ``[[old, new], ...]``. The undo needs both ends to reverse the chain."""

    removed_profile_items_json: str = "[]"
    """``[[bucket, item], ...]`` — the profile items the forget removed.

    Kept in full rather than as ids because a profile item has no id on disk;
    the item itself is what an undo re-merges. Items are small (a sentence and
    its grounding), and a forget removes only the ones a picture alone
    established.
    """

    undone_at: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    """When an undo reversed this forget. Set once; a second undo of the same
    forget is a no-op rather than a second restoration."""
