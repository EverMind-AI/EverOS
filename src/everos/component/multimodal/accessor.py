"""Process-wide multimodal LLM capability accessor.

Lazy singleton — first call reads settings and attempts to build a multimodal
LLM client. Unlike :func:`everos.component.llm.client.get_multimodal_llm_client`
(which raises when misconfigured), this wraps the outcome in a capability that
reports ``available=False`` when the client cannot be built.

Subsequent calls return the cached instance.
"""

from __future__ import annotations

from everos.component.llm.client import (
    LLMNotConfiguredError,
    get_multimodal_llm_client,
)
from everos.core.observability.logging import get_logger

from .capability import MultimodalLLMCapability

logger = get_logger(__name__)

_capability: MultimodalLLMCapability | None = None


def get_multimodal_llm_capability() -> MultimodalLLMCapability:
    """Return the process-wide :class:`MultimodalLLMCapability`. Never raises.

    Lazy singleton: the first call attempts to build a multimodal client from
    current settings — ``available`` is ``False`` when the client cannot be
    built (e.g. missing model/base_url/api_key). Use this from upload
    endpoints, the health endpoint, the startup banner, and any other caller
    that needs to check "is multimodal parsing available?" without a hard
    dependency.

    Build failures (``ValueError`` from the factory / ``LLMNotConfiguredError``
    from missing settings) are logged as
    ``multimodal_llm_capability_build_failed`` so an operator can distinguish
    "not configured" from "misconfigured" — mirrors the equivalent warning on
    the embedding and rerank accessors so all three optional providers surface
    the same signal.
    """
    global _capability
    if _capability is not None:
        return _capability
    try:
        provider = get_multimodal_llm_client()
    except (ValueError, LLMNotConfiguredError) as exc:
        logger.warning(
            "multimodal_llm_capability_build_failed",
            reason=str(exc),
        )
        provider = None
    _capability = MultimodalLLMCapability(provider=provider)
    return _capability
