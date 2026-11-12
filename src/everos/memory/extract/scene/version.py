"""A scene version's fields, computed from its members (docs/scenes.md).

One builder for a user's and a group's scenes: the narrative call's reply
(:class:`~.narrate.SceneText`) over the members it was written for, and the
version's span, attribution and origin computed from those members alone.
A group read compares the span with the asker's windows
(docs/group-memory.md, section 7.7); a user's members carry no attribution,
so the fields stay empty.
"""

from __future__ import annotations

__all__ = ["scene_cluster_kind", "scene_version"]

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from everos.memory.models import ClusterKind, Episode, Origin, Scene
from everos.memory.ports import Scope, Stored

if TYPE_CHECKING:
    from .narrate import SceneText

_CLUSTER_KINDS: dict[str, ClusterKind] = {"user": "user_scene", "group": "group_scene"}


def scene_cluster_kind(scope: Scope) -> ClusterKind:
    """The cluster kind an owner's scene clusters are stored under."""
    return _CLUSTER_KINDS[scope.owner_type]


def scene_version(
    scope: Scope,
    *,
    scene_id: str,
    version: int,
    members: Sequence[Stored[Episode]],
    text: SceneText,
    language: str | None,
) -> Scene:
    """The version ``version`` of ``scene_id`` over ``members`` (oldest first)."""
    items = [m.item for m in members]
    return Scene(
        owner_id=scope.owner_id,
        scene_id=scene_id,
        version=version,
        member_ids=[m.ref.entry_id for m in members],
        subject=text.title,
        summary=text.summary,
        narrative=text.narrative,
        timestamp=max(i.timestamp for i in items),
        source_start=min(
            i.source_start if i.source_start is not None else i.timestamp for i in items
        ),
        sender_ids=_union(i.sender_ids for i in items),
        subject_ids=_union(i.subject_ids for i in items),
        origin=_origin(items),
        language=language,
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
