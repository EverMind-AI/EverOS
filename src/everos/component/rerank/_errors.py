"""Shared error construction for HTTP-based rerank providers."""

from __future__ import annotations

import httpx

from everos.core.observability.logging import get_logger

from .protocol import RerankServiceError

logger = get_logger(__name__)


def upstream_http_error(provider: str, response: httpx.Response) -> RerankServiceError:
    """Log the upstream response body and return a client-safe error.

    The body can carry provider-internal detail, so it is logged rather than
    surfaced in the returned message (which exposes only the status code).

    Args:
        provider: Human-readable provider label (e.g. ``"vLLM"``).
        response: The non-success HTTP response from the rerank backend.

    Returns:
        A :class:`RerankServiceError` naming the provider and status code.
    """
    logger.warning(
        "rerank_http_error",
        provider=provider,
        status=response.status_code,
        body=response.text[:200],
    )
    return RerankServiceError(
        f"{provider} rerank upstream error (HTTP {response.status_code})."
    )


def transport_error(provider: str, exc: Exception) -> RerankServiceError:
    """Build the error for a transport-level failure (connect / timeout)."""
    return RerankServiceError(f"{provider} rerank transport failure: {exc}")


def retries_exhausted_error(provider: str, max_retries: int) -> RerankServiceError:
    """Build the error for a retry loop that fell through without a result."""
    return RerankServiceError(f"{provider} rerank exhausted retries ({max_retries}).")
