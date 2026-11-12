"""``/memory/forget`` and its undo, the picture revision, and the picture read.

``/forget`` takes three bodies (docs/api.md): a user's forget by target
(``user_id`` + ``target``: records by id, a conversation, everything, a kind
and a time range, or a picture), a dry run unless ``dry_run: false``; a
group's (``group_id`` + ``asker_id`` + ``target``, docs/group-memory.md,
section 5.5), a dry run likewise; and the picture body it always took
(``user_id`` + ``image_id``), run at once. A real forget answers with its
``forget_id``, which ``/forget/undo`` takes; the undo also takes a picture's
body (its newest forget). ``/forget/session`` and its undo stay as aliases of
the user's ``session`` target, and ``/images/revise`` of ``/memory/update``.

A forget marks and leaves every version on disk, which is what an undo
depends on: EverOS 2.0 has no hard delete (``/memory/delete`` forgets too).

The ``GET`` route serves a stored picture in one of three copies: the
answering model's (``variant=model``, the copy a search result names as
``model_path``), the original as it arrived, and the decider's thumbnail. It
never changes a memory -- a picture is written once, by ingest; a scaled copy
is a disposable cache entry rendered on first request -- and it never lists
what the store holds, so a reader cannot enumerate another user's pictures by
guessing. A picture group memory refers to is served only to a
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
    UserForgetRequest,
    UserForgetResponseData,
    UserUndoForgetRequest,
    UserUndoForgetResponseData,
)
from everos.service import (
    GroupForgetResult,
    GroupForgetTarget,
    GroupUndoResult,
    UserForgetResult,
    UserForgetTarget,
    UserUndoResult,
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
        403: {
            "model": ErrorResponse,
            "description": "Someone else's personal memory; a group asker who is "
            "not a current member, or not allowed this target",
        },
        404: {
            "model": ErrorResponse,
            "description": "A user's record id that names nothing; or the group "
            "holds no such message, session, member, record or picture",
        },
        500: {
            "model": ErrorResponse,
            "description": "Forget incomplete; send the same request again",
        },
        503: {
            "model": ErrorResponse,
            "description": "A conversation the forget touches is busy; nothing "
            "was forgotten, retry later",
        },
    },
)
async def forget(
    req: ForgetRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[
    UserForgetResponseData | ForgetImageResponseData | GroupForgetResponseData
]:
    """Forget in a user's memory or a group's, reversibly; a dry run by default.

    A body naming ``user_id`` and a ``target`` forgets in the user's memory:
    records by id (``record``), one conversation (``session``), everything
    (``owner``), the memories of some kinds in a time range (``condition``),
    or a picture (``image``). A body naming ``group_id`` forgets inside the
    group (docs/group-memory.md, section 5.5). Both are a **dry run** unless
    ``dry_run`` is false: the answer then says what the forget would touch.
    A real forget answers with what it touched and its ``forget_id``, which
    ``/forget/undo`` takes. The picture body (``user_id`` + ``image_id``) is
    the picture forget, run at once.
    """
    if isinstance(req, UserForgetRequest):
        forgotten = await app.memory.forget(
            app_id=req.app_id,
            project_id=req.project_id,
            user_id=req.user_id,
            asker_id=req.asker_id,
            asserted_asker=asker_header,
            target=_user_target(req),
            reason=req.reason,
            dry_run=req.dry_run,
        )
        return SuccessEnvelope(
            request_id=request_id, data=_user_forget_data(req.user_id, forgotten)
        )
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
        asserted_asker=asker_header,
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
        503: {
            "model": ErrorResponse,
            "description": "The conversation is being memorized; nothing was "
            "forgotten, retry later",
        },
    },
)
async def forget_session(
    req: ForgetSessionRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[ForgetSessionResponseData]:
    """Forget one conversation's memories for one user, at once and reversibly.

    An alias of ``/forget`` with a ``session`` target and ``dry_run: false``:
    the conversation's current episode versions and picture memories are
    marked, and ``/forget/session/undo`` (or ``/forget/undo`` with the
    ``forget_id``) reverses it. A picture file survives while any other live
    memory still references it. A profile item goes once every segment it
    rests on is forgotten.
    """
    result = await app.memory.forget_session(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
        reason=req.reason,
        asserted_asker=asker_header,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=ForgetSessionResponseData(
            session_id=result.session_id,
            forget_id=result.forget_id,
            episodes=list(result.episodes),
            image_memories=result.image_memories,
            images=list(result.images),
            profiles=result.profiles,
        ),
    )


@router.post("/forget/session/undo")
async def undo_forget_session(
    req: UndoForgetSessionRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[UndoForgetSessionResponseData]:
    """Undo the newest forget of one conversation for one user."""
    result = await app.memory.undo_forget_session(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
        asserted_asker=asker_header,
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
            profiles=result.profiles,
        ),
    )


@router.post(
    "/images/revise",
    responses={
        404: {"model": ErrorResponse, "description": "No such picture memory"},
        409: {
            "model": ErrorResponse,
            "description": "The named occurrence was replaced (VERSION_CONFLICT) "
            "or forgotten (MEMORY_FORGOTTEN)",
        },
    },
)
async def revise_image_memory(
    req: ReviseImageMemoryRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[ReviseImageMemoryResponseData]:
    """Write a revised version of one picture memory; the old one is superseded.

    An alias of ``/memory/update`` for a picture memory, by its hash. The
    picture is the same picture: only what the memory says changes. The new
    entry supersedes the old, and the cascade recomputes its keyword index
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
        asserted_asker=asker_header,
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
            "description": "Someone else's personal memory; a group asker not "
            "allowed the forget it reverses",
        },
        404: {"model": ErrorResponse, "description": "No such forget"},
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
async def undo_forget(
    req: UndoRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[
    UserUndoForgetResponseData | UndoForgetResponseData | GroupUndoForgetResponseData
]:
    """Undo a forget: a user's by ``forget_id``, a group's, or a picture's newest.

    A body naming ``user_id`` and ``forget_id`` undoes that forget of the user
    (any target): exactly what it marked comes back, except what another
    live forget names too. A body naming ``group_id`` undoes the group forget
    ``forget_id``, with the forget's own authorisation; 409 when a later
    forget rewrote what it wrote (undo that one first). The picture body
    (``user_id`` + ``image_id``) undoes the picture's newest forget.
    ``applied: false`` means there was nothing to undo (already undone, or
    never forgotten).
    """
    if isinstance(req, UserUndoForgetRequest):
        undone = await app.memory.undo_forget(
            app_id=req.app_id,
            project_id=req.project_id,
            user_id=req.user_id,
            asker_id=req.asker_id,
            asserted_asker=asker_header,
            forget_id=req.forget_id,
        )
        return SuccessEnvelope(
            request_id=request_id, data=_user_undo_data(req.user_id, undone)
        )
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
        asserted_asker=asker_header,
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
        pattern="^(original|model|thumbnail)$",
        description="``model`` for an answering model: the ``[images] model_edge`` "
        "copy (1024 px on the long side by default) a search result names as "
        "``model_path``; ``original`` for the bytes as they arrived; ``thumbnail`` "
        "for the decider-sized copy. Both copies are cached on first request.",
    ),
    asker_id: str | None = Query(
        default=None,
        description="Who asks. Needed for a picture that group memory refers "
        "to: it is served to a member who can see one of those memories.",
    ),
) -> Response:
    """One stored picture, by id, as bytes.

    ``model`` renders at ``[images] model_edge`` and ``cache_quality`` -- the
    copy extraction read the picture at and the one to show an answering model.
    ``thumbnail`` renders at ``[decider].image_thumbnail_edge`` — the same size
    the retrieval loop shows the decider — so a caller can preview exactly what
    it saw. A copy is JPEG, or PNG for a greyscale picture, and is labelled as
    what it is. A picture the asker may not see answers exactly as one that does
    not exist (docs/group-memory.md, section 9.1).
    """
    if not await app.memory.picture_visible(
        image_id, asker_id=asker_id, asserted_asker=asker_header
    ):
        raise HTTPException(status_code=404, detail="no such picture")
    settings = load_settings()
    try:
        if variant != "original":
            if variant == "model":
                edge = settings.images.model_edge
                quality = settings.images.cache_quality
            else:
                edge = settings.decider.image_thumbnail_edge
                quality = settings.decider.image_quality
            data = await app.images.read_derived(image_id, edge=edge, quality=quality)
            return Response(content=data, media_type=_copy_media_type(data))
        data = await app.images.read(image_id)
        info = await app.images.metadata(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="no such picture") from exc
    media_type = _MEDIA_TYPES.get(info.media_type if info else "jpeg", "image/jpeg")
    return Response(content=data, media_type=media_type)


def _copy_media_type(data: bytes) -> str:
    """A scaled copy's media type: PNG for a greyscale picture, else JPEG."""
    return "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"


def _user_target(req: UserForgetRequest) -> UserForgetTarget:
    """The request's target as the service takes it."""
    data = req.target.model_dump()
    return UserForgetTarget(
        kind=data["kind"],
        record_ids=tuple(data.get("record_ids") or ()),
        session_id=data.get("session_id"),
        image_id=data.get("image_id"),
        memory_types=tuple(data.get("memory_types") or ()),
        start=data.get("start"),
        end=data.get("end"),
    )


def _user_forget_data(user_id: str, result: UserForgetResult) -> UserForgetResponseData:
    return UserForgetResponseData(
        forget_id=result.forget_id,
        dry_run=result.dry_run,
        user_id=user_id,
        target={
            key: value
            for key, value in result.target.to_json().items()
            if value not in (None, [], ())
        },
        episodes=list(result.episodes),
        new_episodes=list(result.new_episodes),
        image_memories=list(result.image_memories),
        images=list(result.images),
        scenes=list(result.scenes),
        new_scenes=list(result.new_scenes),
        profile_items=list(result.profile_items),
    )


def _user_undo_data(user_id: str, result: UserUndoResult) -> UserUndoForgetResponseData:
    return UserUndoForgetResponseData(
        forget_id=result.forget_id,
        applied=result.applied,
        user_id=user_id,
        target={
            key: value
            for key, value in result.target.to_json().items()
            if value not in (None, [], ())
        }
        if result.target is not None
        else None,
        episodes=list(result.episodes),
        image_memories=list(result.image_memories),
        images=list(result.images),
        scenes=list(result.scenes),
        profile_items=result.profile_items,
    )


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
