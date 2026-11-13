"""What the memory use cases return to their callers."""

from __future__ import annotations

__all__ = ["FlushResult", "MemorizeResult", "Queued"]

from typing import Literal

from pydantic import BaseModel


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it).

    ``rejected`` and ``failed`` count the batch's non-text items that were
    dropped at ingest -- pictures refused, and items the parser could not
    read (:class:`~everos.memory.IngestResult`); the messages themselves are
    kept. ``duplicates`` counts the batch's messages its group already held,
    which were skipped (docs/group-memory.md, section 5.2).
    """

    message_count: int
    status: Literal["accumulated", "extracted"]
    rejected: int = 0
    failed: int = 0
    duplicates: int = 0


class FlushResult(BaseModel):
    """What flush returns to the caller.

    ``no_extraction`` means the flush carved no cell out: the session buffer
    was empty, or nothing in it formed a memcell.
    """

    status: Literal["extracted", "no_extraction"]


class Queued(BaseModel):
    """A request persisted to the ``async_mode`` queue instead of run inline."""

    message_count: int
    pending: int
    """Unfinished queued requests of the session, this one included."""
