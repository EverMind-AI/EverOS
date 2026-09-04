"""Cascade control — freeze the md → LanceDB projection into a snapshot.

``POST /api/v1/cascade/quiesce`` drains the projection queue and then stops the
cascade subsystem, so the index stops changing until the process restarts.

Why a control endpoint exists at all. Cascade keeps LanceDB in step with the
markdown that is the actual source of truth, and paying for that means rewriting
files: ``merge_insert`` lands each write in a fresh fragment, ``optimize()``
merges the accumulated fragments into one, and ``prune()`` then physically
reclaims the superseded copies -- with a 60s retention window, because the
storage soak measured a 300s window retaining ~24 full-table copies.

Prune's safety argument is that 60s "comfortably" outlives an in-flight read,
which it documents as "sub-second to a few seconds". A read that takes **longer
than the window** breaks it: the reader resolves a version, works, and comes back
to files that were reclaimed underneath it --

    LanceError(IO): Object at location .../_indices/<uuid>/tokens.lance not found

That is not hypothetical. A multi-round retrieval whose decider reads full
episode text spends 56-72s inside one search call (measured), and a benchmark
running ingest and retrieval in the same process lost 45% of one dataset's
questions to exactly this before the endpoint existed: HTTP 500 per search, an
empty context handed to the answer model, and a scored zero.

The fix is not a longer window -- that only moves the race. It is to notice that
a read-only phase creates no new fragments, so there is nothing for optimize or
prune to do, and the whole subsystem is pure risk. ``EVEROS_DISABLE_CASCADE``
already covers a process that starts read-only; this covers the process that
writes first and *becomes* read-only, which an env var read once at startup
cannot express.

Quiesce is deliberately one-way. Restart the process to get the projection back.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from everos.core.observability.logging import get_logger
from everos.entrypoints.api.utils import cascade_orchestrator

logger = get_logger(__name__)

router = APIRouter(prefix="/cascade", tags=["cascade"])


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


@router.post("/quiesce", response_model=QuiesceResponse)
async def quiesce(request: Request) -> QuiesceResponse:
    """Drain the projection queue, then stop watcher + scanner + worker.

    Returns when the index is a complete projection of the markdown on disk and
    nothing further will rewrite it. That ordering is the point: stopping first
    would freeze a *partial* index, which reads as a healthy store that quietly
    fails to recall whatever had not been indexed yet.

    Idempotent -- a second call reports ``quiesced=false`` rather than failing.
    """
    orch = cascade_orchestrator(request)
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "cascade subsystem is not running; nothing to quiesce "
                "(disabled via EVEROS_DISABLE_CASCADE, or this app was built "
                "without the cascade lifespan)"
            ),
        )

    before = await orch.queue_summary()
    # Drain BEFORE stopping. `sync_once` is a full scan + drain, so it also picks
    # up markdown the watcher never saw -- which matters here because the watcher
    # is the half most likely to be off (inotify watches are a per-user kernel
    # resource a shared host can exhaust).
    drained = await orch.sync_once()
    await orch.stop()
    after = await orch.queue_summary()

    logger.info(
        "cascade_quiesced",
        drained=drained,
        pending_before=before.pending,
        pending_after=after.pending,
        failed_permanent=after.failed_permanent,
    )
    if after.pending:
        logger.warning(
            "cascade_quiesce_left_pending",
            pending=after.pending,
            reason="index is not a complete projection of the markdown",
        )
    return QuiesceResponse(
        quiesced=True,
        drained=drained,
        pending_before=before.pending,
        pending_after=after.pending,
        failed_permanent=after.failed_permanent,
    )
