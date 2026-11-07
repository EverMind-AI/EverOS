"""``image_original`` — one row per distinct picture file, with its reference count.

The file itself lives under ``figures/`` and is content-addressed; this table is
what makes "stored once, referenced many times" observable. A row exists for
every picture the store has ever accepted, and ``ref_count`` is how many
image-memory entries point at it. Bytes, width and height are recorded here
rather than re-read from the file so that listing a store costs one query
instead of one decode per picture.

``ref_count`` is maintained by the store, not by a trigger, because every
change to it is already inside a transaction that knows whether the picture is
new (insert, count 1), repeated (increment), or retired (decrement, and the row
keeps the file on disk — undo has to be able to restore it).

A row at ``ref_count == 0`` is a picture no memory points at any more. The file
stays: the design keeps forgotten material on disk so an undo is a status
change, not a restore.
"""

from __future__ import annotations

from sqlalchemy import Index
from sqlmodel import Field

from everos.core.utils.datetime import UtcDatetime, get_utc_now
from everos.infra.persistence.sqlite._base.base import BaseTable, UtcDateTimeColumn


class ImageOriginal(BaseTable, table=True):
    """One stored picture file. PK ``image_id`` (SHA-256 of its bytes)."""

    __tablename__ = "image_original"  # type: ignore[assignment]
    __table_args__ = (
        # Creation order is how the store reports growth and how a sweep can
        # find pictures nothing has referenced since they arrived.
        Index("ix_image_original_seen", "first_seen_at"),
    )

    image_id: str = Field(primary_key=True)
    """SHA-256 hex of the original bytes — the file's name and the id every
    memory, episode body and forget call uses."""

    media_type: str
    """Stored extension: ``png`` / ``jpeg`` / ``webp`` / ``gif``."""

    byte_size: int
    width: int
    height: int
    original_path: str
    """Memory-root-relative path, ``figures/<aa>/<hash>.<ext>``."""

    first_seen_at: UtcDatetime = Field(
        default_factory=get_utc_now, sa_type=UtcDateTimeColumn
    )
    """When this picture first arrived. Every later arrival only bumps
    ``ref_count``; the timestamp is not touched, which is what lets the store
    report distinct-picture growth separately from picture traffic."""

    forgotten_at: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    """When the picture was forgotten. The file stays on disk — a forget is a
    status change on the memories, and undo has to restore them without
    restoring the bytes — so this column, not the file's absence, is what says
    the picture is out of circulation."""

    ref_count: int = Field(default=1)
    """Image-memory entries pointing at this file. Never negative: the store
    decrements only rows it has already incremented for the same memory."""
