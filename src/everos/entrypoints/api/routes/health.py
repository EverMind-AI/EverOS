"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns ``{"status": "ok"}`` with HTTP 200."""
    return {"status": "ok"}
