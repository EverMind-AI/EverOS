"""POST /api/v2/memory/search — hybrid retrieval endpoint.

Thin adapter: validate the request DTO, dispatch to the service layer,
return the envelope verbatim. ``request_id`` is generated inside the
service layer (uniform for OSS + cloud); we trust that value
on the way out.
"""

from __future__ import annotations

from fastapi import APIRouter

from everos.entrypoints.api.dependencies import ApplicationDep
from everos.service import SearchRequest, SearchResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=SearchResponse)
async def post_search(req: SearchRequest, app: ApplicationDep) -> SearchResponse:
    """Hybrid retrieval across the configured memory backends."""
    return await app.memory.search(req)
