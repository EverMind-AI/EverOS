"""The picture routes: forget, its undo, a revision, and the two read-only reads.

Forgetting is its own endpoint rather than a mode of ``/memory/delete``: a
delete removes memories, while a forget marks a picture and everything written
about it, and leaves every version on disk. Collapsing the two would make it
impossible to tell "this never happened" from "this happened and was retracted",
which is exactly the distinction an undo depends on.

The two ``GET`` routes serve the decider's thumbnail and the answering model's
original. They read only — a picture is written once, by ingest — and they
never list what the store holds, so a reader cannot enumerate another user's
pictures by guessing. A picture group memory refers to is served only to a
member who can see one of those memories; to anyone else it is a 404 with the
body of a picture that does not exist.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from everos.config import load_settings
from everos.entrypoints.api.dependencies import (
    ApplicationDep,
    AskerHeaderDep,
    RequestIdDep,
)
from everos.entrypoints.api.schemas import (
    ErrorResponse,
    ForgetImageRequest,
    ForgetImageResponseData,
    ForgetSessionRequest,
    ForgetSessionResponseData,
    ForgottenPictureData,
    ReviseImageMemoryRequest,
    ReviseImageMemoryResponseData,
    SuccessEnvelope,
    UndoForgetRequest,
    UndoForgetResponseData,
    UndoForgetSessionRequest,
    UndoForgetSessionResponseData,
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
            pictures=[
                ForgottenPictureData(
                    entry_id=picture.entry_id,
                    image_id=picture.image_id,
                    name=picture.name,
                )
                for picture in result.pictures
            ],
        ),
    )


@router.post(
    "/forget/session",
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Forget incomplete; send the same request again",
        },
    },
)
async def forget_session(
    req: ForgetSessionRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[ForgetSessionResponseData]:
    """Forget one conversation's memories for one user, reversibly.

    Unlike ``/memory/delete`` this removes nothing: the episode versions and
    picture memories are marked, and ``/forget/session/undo`` reverses it. A
    picture file survives while any other live memory still references it.
    """
    result = await app.memory.forget_session(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
        reason=req.reason,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=ForgetSessionResponseData(
            session_id=result.session_id,
            forget_id=result.forget_id,
            episodes=list(result.episodes),
            image_memories=result.image_memories,
            images=list(result.images),
        ),
    )


@router.post("/forget/session/undo")
async def undo_forget_session(
    req: UndoForgetSessionRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[UndoForgetSessionResponseData]:
    """Undo the newest forget of one conversation for one user."""
    result = await app.memory.undo_forget_session(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=UndoForgetSessionResponseData(
            session_id=result.session_id,
            forget_id=result.forget_id,
            applied=result.applied,
            episodes=list(result.episodes),
            image_memories=result.image_memories,
            images=list(result.images),
        ),
    )


@router.post(
    "/images/revise",
    responses={
        404: {"model": ErrorResponse, "description": "No such picture memory"},
    },
)
async def revise_image_memory(
    req: ReviseImageMemoryRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[ReviseImageMemoryResponseData]:
    """Write a revised version of one picture memory; the old one is superseded.

    The picture is the same picture: only what the memory says changes. The
    new entry supersedes the old, and the cascade recomputes its keyword index
    and vector from the new text.
    """
    result = await app.memory.revise_image_memory(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        image_id=req.image_id,
        description=req.description,
        ocr=req.ocr,
        entry_id=req.entry_id,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=ReviseImageMemoryResponseData(
            image_id=result.image_id,
            entry_id=result.entry_id,
            supersedes=result.supersedes,
            description=result.description,
            ocr=result.ocr,
        ),
    )


@router.post(
    "/forget/undo",
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Undo incomplete; send the same request again",
        },
    },
)
async def undo_forget_image(
    req: UndoForgetRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[UndoForgetResponseData]:
    """Undo the newest forget of one picture for one user.

    ``applied: false`` means there was nothing to undo (never forgotten, or
    already undone); the reported occurrences are still the current state.
    """
    result = await app.memory.undo_forget_image(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        image_id=req.image_id,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=UndoForgetResponseData(
            image_id=result.image_id,
            forget_id=result.forget_id,
            applied=result.applied,
            image_memories=result.image_memories,
            episodes=list(result.episodes),
            profiles=result.profiles,
            pictures=[
                ForgottenPictureData(
                    entry_id=picture.entry_id,
                    image_id=picture.image_id,
                    name=picture.name,
                )
                for picture in result.pictures
            ],
        ),
    )


@router.get("/images/{image_id}")
async def get_image(
    image_id: str,
    app: ApplicationDep,
    asker_header: AskerHeaderDep,
    variant: str = Query(
        default="original",
        pattern="^(original|thumbnail)$",
        description="``original`` for the answering model; ``thumbnail`` for a "
        "decider-sized copy, rendered on first request.",
    ),
    asker_id: str | None = Query(
        default=None,
        description="Who asks. Needed for a picture that group memory refers "
        "to: it is served to a member who can see one of those memories.",
    ),
) -> Response:
    """One stored picture, by id, as bytes.

    ``thumbnail`` renders at ``[decider].image_thumbnail_edge`` — the same size
    the retrieval loop shows the decider — so a caller can preview exactly what
    it saw. A picture the asker may not see answers exactly as one that does not
    exist (docs/group-memory.md, section 9.1).
    """
    if not await app.memory.picture_visible(
        image_id, asker_id=asker_id, asserted_asker=asker_header
    ):
        raise HTTPException(status_code=404, detail="no such picture")
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
