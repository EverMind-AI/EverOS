"""What the memory use cases return to their callers."""

from __future__ import annotations

__all__ = ["FlushResult", "MemorizeResult", "Queued"]

from typing import Literal

from pydantic import BaseModel


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it).

    ``degraded`` means the markdown was written but at least one of those
    files failed to land in the index. ``extracted`` is not used for that
    outcome (#364).
    """

    message_count: int
    status: Literal["accumulated", "extracted", "degraded"]
    failed_permanent: int = 0
    """Files this call wrote whose projection failed and will not be retried."""
    failed_retryable: int = 0
    """Files this call wrote whose projection failed and may be retried."""


class FlushResult(BaseModel):
    """What flush returns to the caller.

    ``no_extraction`` means the flush carved no cell out: the session buffer
    was empty, or nothing in it formed a memcell. ``degraded`` means a cell
    was written to markdown but its index projection failed.
    """

    status: Literal["extracted", "no_extraction", "degraded"]
    failed_permanent: int = 0
    """Files this flush wrote whose projection failed and will not be retried."""
    failed_retryable: int = 0
    """Files this flush wrote whose projection failed and may be retried."""


class Queued(BaseModel):
    """A request persisted to the ``async_mode`` queue instead of run inline."""

    message_count: int
    pending: int
    """Unfinished queued requests of the session, this one included."""
