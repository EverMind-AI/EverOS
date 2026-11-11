"""The picture routes: forget, its undo, a revision, and the two read-only reads.

``/forget`` and ``/forget/undo`` also take a group's body (``group_id``): a
forget inside the group of messages, a member, a session, records or a
picture (docs/group-memory.md, section 5.5), and its undo by ``forget_id``.

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
    ForgetImageResponseData,
    ForgetRequest,
    ForgetSessionRequest,
    ForgetSessionResponseData,
    ForgottenMessageData,
    ForgottenPictureData,
    GroupForgetRequest,
    GroupForgetResponseData,
    GroupUndoForgetRequest,
    GroupUndoForgetResponseData,
    ReviseImageMemoryRequest,
    ReviseImageMemoryResponseData,
    SuccessEnvelope,
    UndoForgetResponseData,
    UndoForgetSessionRequest,
    UndoForgetSessionResponseData,
    UndoRequest,
)
from everos.service import GroupForgetResult, GroupForgetTarget, GroupUndoResult

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
        403: {
            "model": ErrorResponse,
            "description": "Group forget: not a current member, or not allowed "
            "this target",
        },
        404: {
            "model": ErrorResponse,
            "description": "No such picture for this user; or the group holds no "
            "such message, session, member, record or picture",
        },
        500: {
            "model": ErrorResponse,
            "description": "Forget incomplete; send the same request again",
        },
        503: {
            "model": ErrorResponse,
            "description": "Group forget: a session it touches is busy; nothing "
            "was forgotten, retry later",
        },
    },
)
async def forget_image(
    req: ForgetRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[ForgetImageResponseData | GroupForgetResponseData]:
    """Forget one picture of a user, or a target inside a group.

    A body naming ``user_id`` and ``image_id`` forgets the user's picture: its
    memories, its vector, and what referred to it. A body naming ``group_id``
    forgets inside the group (docs/group-memory.md, section 5.5): messages, a
    member, a session, records or a picture, propagated to everything derived
    from them; ``dry_run`` reports what it would touch, and the returned
    ``forget_id`` undoes it.

    Returns what was touched, so a caller can see that the target had a
    memory at all instead of assuming success from an empty 200.
    """
    if isinstance(req, GroupForgetRequest):
        forgotten = await app.memory.forget_in_group(
            app_id=req.app_id,
            project_id=req.project_id,
            group_id=req.group_id,
            asker_id=req.asker_id,
            target=_target(req),
            reason=req.reason,
            dry_run=req.dry_run,
            asserted_asker=asker_header,
        )
        return SuccessEnvelope(
            request_id=request_id,
            data=_group_forget_data(req.group_id, forgotten),
        )
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
        403: {
            "model": ErrorResponse,
            "description": "Group undo: not allowed the forget it reverses",
        },
        404: {"model": ErrorResponse, "description": "No such group forget"},
        409: {
            "model": ErrorResponse,
            "description": "Group undo: a later forget rewrote what this one "
            "wrote; undo that one first",
        },
        500: {
            "model": ErrorResponse,
            "description": "Undo incomplete; send the same request again",
        },
    },
)
async def undo_forget_image(
    req: UndoRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[UndoForgetResponseData | GroupUndoForgetResponseData]:
    """Undo the newest forget of one picture for one user, or a group forget.

    ``applied: false`` means there was nothing to undo (never forgotten, or
    already undone); the reported occurrences are still the current state.
    A body naming ``group_id`` undoes the group forget ``forget_id``, with the
    forget's own authorisation; 409 when a later forget rewrote what it wrote
    (undo that one first).
    """
    if isinstance(req, GroupUndoForgetRequest):
        undone = await app.memory.undo_group_forget(
            app_id=req.app_id,
            project_id=req.project_id,
            group_id=req.group_id,
            asker_id=req.asker_id,
            forget_id=req.forget_id,
            asserted_asker=asker_header,
        )
        return SuccessEnvelope(
            request_id=request_id, data=_group_undo_data(req.group_id, undone)
        )
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


def _target(req: GroupForgetRequest) -> GroupForgetTarget:
    """The request's target as the service takes it."""
    data = req.target.model_dump()
    return GroupForgetTarget(
        kind=data["kind"],
        session_id=data.get("session_id"),
        message_ids=tuple(data.get("message_ids") or ()),
        member_id=data.get("member_id"),
        record_ids=tuple(data.get("record_ids") or ()),
        image_id=data.get("image_id"),
    )


def _group_forget_data(
    group_id: str, result: GroupForgetResult
) -> GroupForgetResponseData:
    return GroupForgetResponseData(
        forget_id=result.forget_id,
        dry_run=result.dry_run,
        group_id=group_id,
        target={
            key: value
            for key, value in vars(result.target).items()
            if value not in (None, ())
        },
        messages=[
            ForgottenMessageData(session_id=session_id, message_id=message_id)
            for session_id, message_id in result.messages
        ],
        episodes=list(result.episodes),
        rewritten=result.rewritten,
        new_episodes=list(result.new_episodes),
        image_memories=list(result.image_memories),
        images=list(result.images),
        scenes=list(result.scenes),
        rewritten_scenes=result.rewritten_scenes,
        new_scenes=list(result.new_scenes),
        member_profiles=list(result.member_profiles),
        member_profile_items=list(result.member_profile_items),
        stale_member_profile_items=list(result.stale_member_profile_items),
        group_profile_items=list(result.group_profile_items),
        stale_group_profile_items=list(result.stale_group_profile_items),
    )


def _group_undo_data(
    group_id: str, result: GroupUndoResult
) -> GroupUndoForgetResponseData:
    return GroupUndoForgetResponseData(
        forget_id=result.forget_id,
        applied=result.applied,
        group_id=group_id,
        messages=[
            ForgottenMessageData(session_id=session_id, message_id=message_id)
            for session_id, message_id in result.messages
        ],
        episodes=list(result.episodes),
        image_memories=list(result.image_memories),
        images=list(result.images),
        scenes=list(result.scenes),
        member_profiles=list(result.member_profiles),
        member_profile_items=list(result.member_profile_items),
        group_profile_items=list(result.group_profile_items),
        withdrawn=list(result.withdrawn),
    )
