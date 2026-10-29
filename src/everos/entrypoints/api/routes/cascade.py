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
prune to do, and the whole subsystem is pure risk. ``[cascade] enabled = false``
already covers a process that starts read-only; this covers the process that
writes first and *becomes* read-only, which a setting read once at startup
cannot express.

Quiesce is deliberately one-way. Restart the process to get the projection back.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from everos.entrypoints.api.dependencies import ApplicationDep
from everos.entrypoints.api.schemas import QuiesceResponse

router = APIRouter(prefix="/cascade", tags=["cascade"])


@router.post("/quiesce", response_model=QuiesceResponse)
async def quiesce(app: ApplicationDep) -> QuiesceResponse:
    """Drain the projection queue, then stop watcher + scanner + worker.

    Returns when the index is a complete projection of the markdown on disk and
    nothing further will rewrite it. That ordering is the point: stopping first
    would freeze a *partial* index, which reads as a healthy store that quietly
    fails to recall whatever had not been indexed yet.

    Idempotent -- a second call reports ``quiesced=false`` rather than failing.
    """
    result = await app.ops.cascade_quiesce()
    if result is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "cascade subsystem is not running; nothing to quiesce "
                "(disabled via [cascade] enabled = false, or the application "
                "was not started)"
            ),
        )
    return QuiesceResponse(
        quiesced=True,
        drained=result.drained,
        pending_before=result.pending_before,
        pending_after=result.pending_after,
        failed_permanent=result.failed_permanent,
    )
