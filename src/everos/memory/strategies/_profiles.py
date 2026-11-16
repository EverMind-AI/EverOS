"""A person's profile in a space, brought up to date: one core, two strategies.

A user's profile in their own space (``users/<u>/user.md``,
``extract_user_profile``) and a member's in a group
(``groups/<g>/members/<m>.md``, ``update_member_profiles``) are one document
updated one way; the strategies differ only in when they update it and what
they hand it:

- **Cadence.** A one-to-one space reads its segments on ``[profile]
  update_every``; a group either queues each member's segments and reads
  them every ``[groups] member_profile_every`` (``per_member``), or queues
  the group's segments and reads every member who spoke in a batch with one
  call (``batched``, the default: :mod:`._member_batch`).
- **Input.** The segments to read, as the model is shown them -- a group's
  with its speakers labelled by their roster names -- and the language the
  space resolved.

The update itself is the same sequence for both (:func:`update_profile`): the
pictures the person posted in those segments (none a live forget names),
everalgo's ``ProfileExtractor.aupdate`` (the two-prompt update, then a
consolidation only for a part over its cap that has grown), every item
stamped with its sources and versions
(:func:`~everos.memory.extract.pipeline.profile_items.stamp_update`), what a
person holds kept as they left it
(:func:`~everos.memory.extract.pipeline.profile_items.keep_held`), and the
replaced items moved to the bounded history. A batched group update calls
everalgo's ``ProfileExtractor.aupdate_many`` once for every member instead,
and keeps each member's result the same way (:func:`finish_update`).
"""

from __future__ import annotations

__all__ = ["ProfileUpdated", "finish_update", "update_profile"]

from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile import ProfileUpdate

from everos.memory.deps import MemoryDeps
from everos.memory.extract.pipeline.pictures import profile_pictures
from everos.memory.extract.pipeline.profile_items import (
    bounded_history,
    keep_held,
    stamp_update,
)
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import MemCell, Profile
from everos.memory.ports import Scope


@dataclass(frozen=True)
class ProfileUpdated:
    """One update's result: the document to write, and what it took.

    Attributes:
        profile: The new profile, its history bounded; nothing pending.
        pictures: The pictures the model was shown.
        retired: Stored items the update took out.
        replaced: History entries the update added.
        superseded: Of those, the values a new one replaced.
        consolidated: Parts the update consolidated.
        held: Ids of held items whose change the update undid.
    """

    profile: Profile
    pictures: int
    retired: int
    replaced: int
    superseded: int
    consolidated: int
    held: list[str]


async def update_profile(
    deps: MemoryDeps,
    scope: Scope,
    *,
    person_id: str,
    old: Profile | None,
    history: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[str, MemCell]],
    shown: Sequence[MemCell],
    language: str | None,
    reads: AbstractAsyncContextManager[Any] | None = None,
    pictures: Collection[str] | None = None,
    picture_text: Mapping[str, str] | None = None,
) -> ProfileUpdated:
    """Read ``segments`` into the person's profile; the caller holds its lock.

    Args:
        deps: The model, the picture store and the forget log.
        scope: The space the profile belongs to.
        person_id: Whom the profile is about.
        old: The profile the model updates; ``None`` writes the first one.
        history: The stored history the replaced items join.
        segments: ``(memcell_id, cell)`` read, oldest first: each item's
            sources and span are read off them.
        shown: The same segments as the model is shown them.
        language: The language the profile is written in.
        reads: Held while the update reads storage (the pictures and the
            forget log), never across the model call: a group's flush bounds
            how many members read at once with it.
        pictures: Attach only these of the person's pictures, by content
            hash (:func:`~everos.memory.extract.pipeline.profile_input.
            select_profile_input`); ``None`` attaches every one they posted.
        picture_text: Attach each picture as this text instead of its pixels.
    """
    async with reads or nullcontext():
        cells, pictures = await profile_pictures(
            shown,
            person_id,
            images=deps.stores.images,
            limits=deps.picture_limits,
            forgotten=await deps.stores.forget_log.live_targets(
                target="image",
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type=scope.owner_type,
            ),
            only=pictures,
            as_text=picture_text,
        )
    update = await ProfileExtractor(llm=deps.llm).aupdate(
        cells,
        sender_id=person_id,
        old_profile=profile_mapper.to_algo(old) if old is not None else None,
        segment_ids=[memcell_id for memcell_id, _ in segments],
        pictures=pictures,
        custom_instructions=deps.instructions.load("profile_extract"),
        output_language=language,
    )
    return finish_update(
        old,
        update,
        history=history,
        segments=segments,
        language=language,
        pictures=len(pictures),
    )


def finish_update(
    old: Profile | None,
    update: ProfileUpdate,
    *,
    history: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[str, MemCell]],
    language: str | None,
    pictures: int,
) -> ProfileUpdated:
    """What EverOS keeps of one person's update: every item stamped with its
    sources and versions, what a person holds kept as they left it, the
    replaced items moved to the bounded history.

    Args:
        old: The profile the update was given; ``None`` for a first one.
        update: What everalgo returned for the person.
        history: The stored history the replaced items join.
        segments: ``(memcell_id, cell)`` the person was read from, oldest
            first: each item's span is read off them.
        language: The language the profile is written in.
        pictures: How many pictures of theirs the model was shown.
    """
    stamped, replaced = stamp_update(
        old,
        update,
        cells=dict(segments),
        replaced_at=segments[-1][1].timestamp,
        primary_language=language,
    )
    stamped, replaced, held = keep_held(old, stamped, replaced)
    stamped.history = bounded_history(replaced, history, stamped)
    return ProfileUpdated(
        profile=stamped,
        pictures=pictures,
        retired=len(update.retired),
        replaced=len(replaced),
        superseded=sum(1 for entry in replaced if "superseded_by" in entry),
        consolidated=len(update.consolidated),
        held=held,
    )
