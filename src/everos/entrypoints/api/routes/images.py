"""``POST /api/v2/memory/forget`` and the two read-only picture routes.

Forgetting is its own endpoint rather than a mode of ``/memory/delete``: a
delete removes memories, while a forget marks a picture and everything written
about it, and leaves every version on disk. Collapsing the two would make it
impossible to tell "this never happened" from "this happened and was retracted",
which is exactly the distinction an undo depends on.

The two ``GET`` routes serve the decider's thumbnail and the answering model's
original. They read only — a picture is written once, by ingest — and they
never list what the store holds, so a reader cannot enumerate another user's
pictures by guessing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from everos.config import load_settings
from everos.entrypoints.api.dependencies import ApplicationDep, RequestIdDep
from everos.entrypoints.api.schemas import (
    ErrorResponse,
    ForgetImageRequest,
    ForgetImageResponseData,
    SuccessEnvelope,
)

router = APIRouter(prefix="/memory", tags=["memory"])

_MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


@router.post(
    "/forget",
    responses={
        404: {"model": ErrorResponse, "description": "No such picture for this user"},
        500: {
            "model": ErrorResponse,
            "description": "Forget incomplete; send the same request again",
        },
    },
)
async def forget_image(
    req: ForgetImageRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[ForgetImageResponseData]:
    """Forget one picture: its memories, its vector, and what referred to it.

    Returns what was touched, so a caller can see that the picture had a
    memory at all instead of assuming success from an empty 200.
    """
    result = await app.memory.forget_image(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        image_id=req.image_id,
        reason=req.reason,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=ForgetImageResponseData(
            image_id=result.image_id,
            image_memories=result.image_memories,
            episodes=list(result.episodes),
            profiles=result.profiles,
        ),
    )


@router.get("/images/{image_id}")
async def get_image(
    image_id: str,
    app: ApplicationDep,
    variant: str = Query(
        default="original",
        pattern="^(original|thumbnail)$",
        description="``original`` for the answering model; ``thumbnail`` for a "
        "decider-sized copy, rendered on first request.",
    ),
) -> Response:
    """One stored picture, by id, as bytes.

    ``thumbnail`` renders at ``[decider].image_thumbnail_edge`` — the same size
    the retrieval loop shows the decider — so a caller can preview exactly what
    it saw.
    """
    settings = load_settings()
    try:
        if variant == "thumbnail":
            data = await app.images.read_derived(
                image_id,
                edge=settings.decider.image_thumbnail_edge,
                quality=settings.decider.image_quality,
            )
            return Response(content=data, media_type="image/jpeg")
        data = await app.images.read(image_id)
        info = await app.images.metadata(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="no such picture") from exc
    media_type = _MEDIA_TYPES.get(info.media_type if info else "jpeg", "image/jpeg")
    return Response(content=data, media_type=media_type)
