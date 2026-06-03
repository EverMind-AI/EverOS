"""Prometheus metrics route."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

from everos.core.observability.metrics import generate_metrics_response

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    """Render the current Prometheus registry in exposition format."""
    return Response(
        content=generate_metrics_response(),
        media_type=CONTENT_TYPE_LATEST,
    )
