"""What the memory use cases return to their callers."""

from __future__ import annotations

__all__ = ["MemorizeResult", "Queued"]

from typing import Literal

from pydantic import BaseModel


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it)."""

    message_count: int
    status: Literal["accumulated", "extracted"]


class Queued(BaseModel):
    """A request persisted to the ``async_mode`` queue instead of run inline."""

    message_count: int
    pending: int
    """Unfinished queued requests of the session, this one included."""
