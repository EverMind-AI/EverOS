"""POST /api/v2/memory/search — multi-round retrieval endpoint.

Thin adapter: validate the request DTO, dispatch to the service layer,
return the envelope verbatim. ``request_id`` is generated inside the
service layer (uniform for OSS + cloud); we trust that value
on the way out.
"""

from __future__ import annotations

from fastapi import APIRouter

from everos.entrypoints.api.dependencies import ApplicationDep, AskerHeaderDep
from everos.service import SearchRequest, SearchResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=SearchResponse)
async def post_search(
    req: SearchRequest, app: ApplicationDep, asker_header: AskerHeaderDep
) -> SearchResponse:
    """Multi-round retrieval over a user's or a group's memory.

    ``method`` other than ``llm_multiround`` and an ``agent_id`` (EverOS 2.0 has
    no agent memory) are refused (422). A group scope (``group_id`` /
    ``group_ids``) needs an ``asker_id`` who is a current member; it returns
    only what lies inside the asker's membership windows (403 for a
    non-member).
    """
    return await app.memory.search(req, asserted_asker=asker_header)
