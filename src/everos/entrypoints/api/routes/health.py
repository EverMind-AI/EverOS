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


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe with capabilities and disabled features."""
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
    )
