"""POST /api/v2/memory/search — hybrid retrieval endpoint.

Thin adapter: validate the request DTO, dispatch to the service layer,
return the envelope verbatim. ``request_id`` is generated inside the
:class:`SearchManager` (uniform for OSS + cloud); we trust that value
on the way out.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from everos.entrypoints.api.utils import application
from everos.memory.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=SearchResponse)
async def post_search(req: SearchRequest, request: Request) -> SearchResponse:
    """Hybrid retrieval across the configured memory backends."""
    return await application(request).memory.search(req)
