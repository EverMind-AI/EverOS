"""Shared helpers for the API layer (routes + exception handlers)."""

from __future__ import annotations

from fastapi import Request

from everos.core.observability.tracing import gen_request_id


def extract_request_id(request: Request) -> str:
    """Return the request_id set by middleware, or mint a fresh fallback."""
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else gen_request_id()
