"""POST /api/v1/memory/search — hybrid retrieval endpoint.

Thin adapter: validate the request DTO, dispatch to the service layer,
return the envelope verbatim. ``request_id`` is generated inside the
:class:`SearchManager` (uniform for OSS + cloud); we trust that value
on the way out.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from everos.memory.search import FilterError, SearchRequest, SearchResponse
from everos.service import search

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.post("/search", response_model=SearchResponse)
async def post_search(req: SearchRequest) -> SearchResponse:
    """Hybrid retrieval across the configured memory backends."""
    try:
        return await search(req)
    except FilterError as exc:
        # Filter-DSL violations surface as 422 with the compile message
        # (mirrors /get's contract).
        raise HTTPException(status_code=422, detail=str(exc)) from exc
