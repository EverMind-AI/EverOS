"""Response model for ``POST /cascade/quiesce``."""

from __future__ import annotations

from pydantic import BaseModel


class QuiesceResponse(BaseModel):
    """What the final drain accomplished, and what it left behind.

    ``pending_before`` is the queue depth on arrival: a caller that expected the
    projection to be up to date can assert it was 0 and learn otherwise. Anything
    above 0 in ``pending_after`` means the drain could not finish -- the index is
    NOT a complete projection of the markdown, and searching it will silently
    under-recall.
    """

    quiesced: bool
    """False when the subsystem was already stopped or never started; the call
    is idempotent, so this is information rather than an error."""
    drained: int
    """Rows the final scan + drain cycle processed."""
    pending_before: int
    pending_after: int
    failed_permanent: int
    """Files that need ``cascade fix``; a data-quality backlog, not a drain
    failure. Reported so a caller can see that some markdown never made it."""
