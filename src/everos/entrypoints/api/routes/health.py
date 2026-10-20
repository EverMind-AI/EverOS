"""Health check route."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from everos import __version__
from everos.component.capabilities import compute_disabled_features
from everos.component.embedding import get_embedding_capability
from everos.component.multimodal import get_multimodal_llm_capability
from everos.component.parser import parser_available
from everos.component.rerank import get_rerank_capability
from everos.core.observability.logging import get_logger
from everos.entrypoints.api.utils import cascade_orchestrator, memorize_queue
from everos.infra.persistence.index import last_probe

logger = get_logger(__name__)

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
    """Present when the cascade lifespan is running; ``None`` for a
    minimal app built without it."""
    memorize_queue: MemorizeQueueHealthBlock | None = None
    """Present when the memorize queue lifespan is running."""


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
    # ``llm`` is hardcoded ``True`` — kept for symmetry with the caps
    # dict rather than probed live. Rationale: LLM is a Tier-1 hard
    # requirement enforced at startup by ``LLMLifespanProvider``
    # (lifespans/llm.py), which eagerly calls ``get_llm_client()`` and
    # raises ``LLMNotConfiguredError`` if credentials are missing —
    # FastAPI startup then fails, so ``/health`` is unreachable
    # without a working LLM. Any code path that reaches this handler
    # therefore has ``get_llm_client()`` returning a real client. If
    # the LLM capability is ever downgraded to soft (like embed /
    # rerank), swap this literal for a real probe.
    caps = HealthCapabilities(
        llm=True,
        embed=last_probe() is not None,
        rerank=get_rerank_capability().available,
        multimodal_llm=get_multimodal_llm_capability().available,
        parser=parser_available(),
    )
    cascade: CascadeHealthBlock | None = None
    orch = cascade_orchestrator(request)
    if orch is not None:
        try:
            ch = await orch.health()
            cascade = CascadeHealthBlock(
                healthy=ch.healthy,
                reasons=ch.reasons,
                pending=ch.pending,
                failed_permanent=ch.failed_permanent,
                failed_retryable=ch.failed_retryable,
                drain_consecutive_failures=ch.drain_consecutive_failures,
                unrecoverable_total=ch.unrecoverable_total,
                optimize_failure_streak=ch.optimize_failure_streak,
                prune_stale_seconds=round(ch.prune_stale_seconds, 1),
                index_rows=ch.index_rows,
                index_emptied=ch.index_emptied,
            )
        except Exception as exc:
            # The probe reads SQLite (queue_summary runs aggregate counts).
            # A locked / full / mid-migration DB must NOT turn /health into a
            # 500 — that flips the liveness signal and makes k8s restart the
            # container, which fixes neither a stuck DB nor disk bloat (see
            # the handler docstring). Surface it as unhealthy *readiness*
            # with a reason and keep HTTP 200.
            logger.warning("cascade_health_probe_failed", error=repr(exc))
            cascade = CascadeHealthBlock(
                healthy=False,
                reasons=[f"cascade health probe failed: {exc!r}"],
                pending=0,
                failed_permanent=0,
                failed_retryable=0,
                drain_consecutive_failures=0,
                unrecoverable_total=0,
                optimize_failure_streak=0,
                prune_stale_seconds=0.0,
            )
    queue_block: MemorizeQueueHealthBlock | None = None
    worker = memorize_queue(request)
    if worker is not None:
        try:
            qh = await worker.health()
            queue_block = MemorizeQueueHealthBlock(**qh.model_dump())
        except Exception as exc:
            # Same rule as the cascade block: a locked DB must not turn the
            # liveness probe into a 500.
            logger.warning("memorize_queue_health_probe_failed", error=repr(exc))
            queue_block = MemorizeQueueHealthBlock(
                healthy=False,
                reasons=[f"memorize queue health probe failed: {exc!r}"],
                loop_alive=False,
                accepting=False,
                pending=0,
                oldest_pending_age_s=None,
                failed=0,
                loop_restarts=0,
                last_loop_error_at=None,
            )
    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=caps,
        # `embed` reports the startup probe, but `disabled_features` is a
        # client contract about what the server will actually refuse -- and
        # every engine-side gate (search, the embed-dependent strategies,
        # backfill) still asks whether a provider could be built. A service
        # that was unreachable at startup and came back would otherwise have
        # the server announcing vector search disabled while serving it.
        disabled_features=compute_disabled_features(
            {**caps.model_dump(), "embed": get_embedding_capability().available}
        ),
        cascade=cascade,
        memorize_queue=queue_block,
    )
