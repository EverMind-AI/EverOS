"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from everos import __version__
from everos.component.capabilities import compute_disabled_features
from everos.component.embedding import get_embedding_capability
from everos.component.multimodal import get_multimodal_llm_capability
from everos.component.parser import parser_available
from everos.component.rerank import get_rerank_capability
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import md_change_state_repo

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class HealthCapabilities(BaseModel):
    """Availability flags for the five capability probes.

    Field order matches the health-endpoint payload contract; clients
    key off these names to decide whether to expose optional features.
    """

    llm: bool
    embed: bool
    rerank: bool
    multimodal_llm: bool
    parser: bool


class HealthCascade(BaseModel):
    """Cascade queue counters, mirroring ``cascade status`` (#364).

    Writes land in markdown first and are indexed later by the cascade
    worker, so a successful ``/memory/flush`` says nothing about whether
    the row reached the index. When the worker is stuck, the write path
    keeps returning ``extracted`` while nothing is queryable — the outage
    stays invisible to an HTTP host until someone reads the server log.

    These counters already existed for the CLI; exposing them here gives
    an API host the same probe the CLI has. ``failed_permanent > 0``
    means rows are terminally stuck: no retry will clear them, and the
    operator has to run ``cascade fix`` / ``cascade rebuild``.
    """

    pending: int
    done: int
    failed_retryable: int
    failed_permanent: int
    lag: int
    """``max_lsn - last_processed_lsn`` — how far indexing trails writes."""


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``.

    Declared as a Pydantic model (not ``dict``) so the generated
    OpenAPI schema carries the full field shape — ``capabilities`` and
    ``disabled_features`` are typed. A bare ``-> dict`` return type
    degrades the OpenAPI response to ``additionalProperties: true``,
    which robs clients (and codegen) of any structure to lean on.
    """

    status: str
    version: str
    capabilities: HealthCapabilities
    disabled_features: list[str]
    cascade: HealthCascade | None = None
    """``None`` when queue state is unreadable — see ``_cascade_health``."""


async def _cascade_health() -> HealthCascade | None:
    """Queue counters, or ``None`` if they cannot be read.

    ``/health`` is a liveness probe: it must answer even when the
    metadata store is unavailable, so a failure here degrades the
    payload instead of the endpoint. ``None`` is deliberately distinct
    from all-zero counters — "unknown" and "queue is clean" are
    different facts, and a host that treats them alike would report a
    dead queue as healthy.
    """
    try:
        summary = await md_change_state_repo.queue_summary()
    except Exception:
        logger.warning("health.cascade.unavailable", exc_info=True)
        return None
    return HealthCascade(
        pending=summary.pending,
        done=summary.done,
        failed_retryable=summary.failed_retryable,
        failed_permanent=summary.failed_permanent,
        lag=max(0, summary.max_lsn - summary.last_processed_lsn),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe with capabilities, disabled features and queue state."""
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
        embed=get_embedding_capability().available,
        rerank=get_rerank_capability().available,
        multimodal_llm=get_multimodal_llm_capability().available,
        parser=parser_available(),
    )
    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=caps,
        disabled_features=compute_disabled_features(caps.model_dump()),
        cascade=await _cascade_health(),
    )
