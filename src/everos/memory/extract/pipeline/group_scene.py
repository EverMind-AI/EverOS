"""A group scene version: its narrative over member episodes, and its fields.

docs/group-memory.md, section 7.7. Shared by the scene strategy, which writes
a version each time an episode joins a scene, and by a forget inside a group,
which writes one when an episode of the scene was re-extracted or forgotten
(section 9.4): the narrative is everalgo's ``EpisodeReflector`` -- INIT over
the members, or UPDATE of the previous narrative with the new ones -- and the
version's span, attribution and origin are computed from its members alone.

A forget always narrates from scratch (INIT): the previous narrative may say
what the forgotten message said, so nothing of it may be carried over.
"""

from __future__ import annotations

__all__ = ["narrate_scene", "scene_version"]

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.user_memory import EpisodeReflector

from everos.memory.models import Episode, Origin, Scene
from everos.memory.ports import Scope, Stored

if TYPE_CHECKING:
    from everalgo.llm import LLMClient


async def narrate_scene(
    llm: LLMClient,
    *,
    owner_id: str,
    members: Sequence[Stored[Episode]],
    previous: Scene | None,
    custom_instructions: str | None,
    language: str | None,
) -> AlgoEpisode:
    """The scene's narrative: INIT over ``members``, or UPDATE of ``previous``.

    With ``previous``, ``members`` are the episodes the previous version does
    not cover yet (the update adds them); without it, every member.
    """
    reflector = EpisodeReflector(llm=llm)
    if previous is None:
        return await reflector.areflect(
            [_algo(m) for m in members],
            custom_instructions=custom_instructions,
            output_language=language,
        )
    return await reflector.areflect(
        [_algo(m) for m in members],
        old_episode=AlgoEpisode(
            owner_id=owner_id,
            episode=previous.narrative,
            subject=previous.subject,
            summary=previous.summary,
            timestamp=previous.timestamp,
        ),
        custom_instructions=custom_instructions,
        output_language=language,
    )


def scene_version(
    scope: Scope,
    *,
    scene_id: str,
    version: int,
    members: Sequence[Stored[Episode]],
    narrative: AlgoEpisode,
    language: str | None,
) -> Scene:
    """The version ``version`` of ``scene_id`` over ``members`` (oldest first)."""
    items = [m.item for m in members]
    return Scene(
        owner_id=scope.owner_id,
        scene_id=scene_id,
        version=version,
        member_ids=[m.ref.entry_id for m in members],
        subject=narrative.subject,
        summary=narrative.summary,
        narrative=narrative.episode,
        timestamp=max(i.timestamp for i in items),
        source_start=min(
            i.source_start if i.source_start is not None else i.timestamp for i in items
        ),
        sender_ids=_union(i.sender_ids for i in items),
        subject_ids=_union(i.subject_ids for i in items),
        origin=_origin(items),
        language=language,
    )


def _algo(stored: Stored[Episode]) -> AlgoEpisode:
    episode = stored.item
    return AlgoEpisode(
        owner_id=episode.owner_id,
        episode=episode.episode,
        subject=episode.subject,
        summary=episode.summary,
        timestamp=episode.timestamp,
    )


def _union(lists: Iterable[Sequence[str]]) -> list[str]:
    out: list[str] = []
    for values in lists:
        for value in values:
            if value not in out:
                out.append(value)
    return out


def _origin(episodes: Sequence[Episode]) -> Origin | None:
    """``member`` / ``assistant`` when every member episode agrees, else ``mixed``."""
    origins = {e.origin for e in episodes if e.origin is not None}
    if not origins:
        return None
    if len(origins) == 1:
        (origin,) = origins
        return origin
    return "mixed"
