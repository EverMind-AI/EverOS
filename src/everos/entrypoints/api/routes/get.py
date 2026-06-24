"""POST /api/v1/memory/get — paginated listing endpoint.

Thin adapter: validate the request DTO, dispatch to the service layer,
return the envelope verbatim. ``request_id`` is generated inside the
:class:`GetManager`; we trust the value on the way out.
"""

from __future__ import annotations

from fastapi import APIRouter

from everos.memory.get import GetRequest, GetResponse
from everos.service import get as get_service

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.post("/get", response_model=GetResponse)
async def post_get(req: GetRequest) -> GetResponse:
    """Paginated listing over the requested ``memory_type``."""
    return await get_service(req)
