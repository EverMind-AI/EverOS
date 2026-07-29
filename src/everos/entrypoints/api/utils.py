"""Shared helpers for the API layer (routes + exception handlers)."""

from __future__ import annotations

from fastapi import Request

from everos.core.observability.tracing import gen_request_id
from everos.memory.cascade import CascadeOrchestrator


def extract_request_id(request: Request) -> str:
    """Return the request_id set by middleware, or mint a fresh fallback."""
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else gen_request_id()


def cascade_orchestrator(request: Request) -> CascadeOrchestrator | None:
    """Return the running cascade orchestrator, or ``None``.

    The cascade lifespan stashes the orchestrator at
    ``app.state.lifespan_data["cascade"]``. An app built without that
    lifespan (e.g. a minimal test app) has no entry, so callers get
    ``None`` and degrade gracefully instead of erroring.
    """
    data = getattr(request.app.state, "lifespan_data", None) or {}
    orch = data.get("cascade")
    return orch if isinstance(orch, CascadeOrchestrator) else None
