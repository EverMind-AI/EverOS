"""Health check route."""

from __future__ import annotations

import dataclasses
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from everos import __version__
from everos.entrypoints.api.utils import application

router = APIRouter(tags=["health"])


class HealthCapabilities(BaseModel):
    """Availability flags for the five capability probes.

    Field order matches the health-endpoint payload contract; clients
    key off these names to decide whether to expose optional features.
    """

    llm: bool
    embed: bool = Field(
        description=(
            "Whether the embedding service answered the startup dimension "
            "probe. A snapshot taken when the process started, not a "
            "liveness check: a service that recovers later still reads false "
            "until the next restart."
        )
    )
    """Whether the embedding service answered the dimension probe **at startup**.

    A snapshot, not a liveness check: a service that recovers after the
    process started still reads ``false`` until the next restart, because
    re-probing here would put a network call behind an endpoint meant to be
    polled. Before, this reported whether a provider could be *constructed*,
    which never touches the network -- so an unreachable embedding service
    read ``true`` while every write failed (issue #434).
    """
    rerank: bool
    multimodal_llm: bool
    parser: bool


class CascadeHealthBlock(BaseModel):
    """Readiness of the md → LanceDB projection (cascade) subsystem.

    ``healthy`` reflects **operational** health only — drain loop alive,
    optimize not stuck, version cleanup (prune) not stalled — and is what
    alerting should watch. ``failed_permanent`` (md files awaiting
    ``cascade fix``) is a normal data-quality backlog reported as an
    informational count; it does **not** flip ``healthy``, otherwise the
    signal would sit red forever.
    """

    healthy: bool
    reasons: list[str]
    pending: int
    failed_permanent: int
    failed_retryable: int
    drain_consecutive_failures: int
    unrecoverable_total: int
    optimize_failure_streak: int
    prune_stale_seconds: float
    index_rows: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Rows per business index table, counted once when the process "
            "started. A snapshot, not a live count: counting costs a round "
            "trip per collection on a remote backend and this endpoint is "
            "polled. `everos cascade status` counts live."
        ),
    )
    index_emptied: bool = Field(
        default=False,
        description=(
            "True when the cascade ledger says every file is indexed, "
            "nothing is pending, and every index table is empty -- the "
            "state left behind by deleting the index directory, which the "
            "ledger in .index/sqlite survives. Searches return nothing "
            "while every other signal here reads healthy. Recover with "
            "`everos cascade rebuild`; markdown is untouched. Informational "
            "like failed_permanent: it does not flip `healthy`, because it "
            "is a startup snapshot that could not clear until a restart."
        ),
    )


class MemorizeQueueHealthBlock(BaseModel):
    """Readiness of the ``async_mode`` memorize queue.

    ``healthy`` is the delivery loop being alive. The probe only observes;
    the loop relaunches itself (and on the next ``/add``), and
    ``loop_restarts`` counts how often that happened. ``pending`` and
    ``oldest_pending_age_s`` are the backlog: growing with a live loop means
    delivery is slower than arrival (or the LLM is down), which is a
    capacity signal, not a liveness one. Alert on ``healthy`` and on
    ``loop_restarts`` changing.
    """

    healthy: bool
    reasons: list[str]
    loop_alive: bool
    accepting: bool
    pending: int
    oldest_pending_age_s: float | None
    failed: int
    loop_restarts: int
    last_loop_error_at: datetime | None


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``.

    Declared as a Pydantic model (not ``dict``) so the generated
    OpenAPI schema carries the full field shape — ``capabilities``,
    ``disabled_features`` and ``cascade`` are typed. A bare ``-> dict``
    return type degrades the OpenAPI response to
    ``additionalProperties: true``, which robs clients (and codegen) of
    any structure to lean on.
    """

    status: str
    version: str
    capabilities: HealthCapabilities
    disabled_features: list[str]
    cascade: CascadeHealthBlock | None = None
    """Present while the started application runs the cascade; ``None``
    otherwise (disabled, or the application was never started)."""
    memorize_queue: MemorizeQueueHealthBlock | None = None
    """Present while the started application runs the memorize queue."""


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + capabilities + cascade readiness probe.

    ``status`` stays ``"ok"`` whenever the process is up — the HTTP code
    is a *liveness* signal and a degraded cascade must not trigger a
    restart (a crash-loop fixes neither a bad md file nor disk bloat).
    The ``cascade`` block is the *readiness* signal: ``healthy=false``
    with human-readable ``reasons`` **only** when the projection pipeline
    itself is stuck (drain failing, optimize stuck, version cleanup
    stalled). ``failed_permanent`` — files awaiting ``cascade fix`` — is
    a data-quality backlog reported as an informational count that does
    not flip ``healthy``. Alert on ``cascade.healthy``.
    """
    report = await application(request).ops.health()
    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=HealthCapabilities(**dataclasses.asdict(report.providers)),
        disabled_features=report.disabled_features,
        cascade=(
            None
            if report.cascade is None
            else CascadeHealthBlock(**dataclasses.asdict(report.cascade))
        ),
        memorize_queue=(
            None
            if report.memorize_queue is None
            else MemorizeQueueHealthBlock(**report.memorize_queue.model_dump())
        ),
    )
