"""POST /api/v2/memory/get — paginated listing endpoint.

Thin adapter: validate the request DTO, dispatch to the service layer,
return the envelope verbatim. ``request_id`` is generated inside the
:class:`GetManager`; we trust the value on the way out.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from everos.entrypoints.api.utils import application
from everos.memory.get import GetRequest, GetResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/get", response_model=GetResponse)
async def post_get(req: GetRequest, request: Request) -> GetResponse:
    """Paginated listing over the requested ``memory_type``."""
    return await application(request).memory.get(req)
