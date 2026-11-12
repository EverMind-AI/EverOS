"""Group-profile items on the read side: shaping and the visible current profile.

docs/group-memory.md, sections 5.3, 5.4 and 8.5. Every read goes through the
index with the read scope's predicate in the ``where`` -- the windows decide
in the query, never afterwards -- so an item whose sources lie outside the
asker's windows is absent, exactly as if it did not exist:

* :func:`visible_profile_items` -- the current items (``predicate`` excludes
  superseded and forgotten ones), folded to the current profile: the newest
  version of each topic, current decisions, open dissent, active roles;
* :func:`item_versions` -- every version but the forgotten
  (``include_superseded``), through the scope's ``history_predicate``.

A fold over what the asker can see may differ from the group's: a topic whose
newest version lies outside the asker's windows is not shown at all, never
in an older version that is no longer true.
"""

from __future__ import annotations

__all__ = [
    "item_versions",
    "shape_profile_item",
    "visible_profile",
    "visible_profile_items",
]

from collections.abc import Sequence

from everos.memory.access import ReadScope
from everos.memory.extract.group_profile import GroupProfileView, current_profile
from everos.memory.models import GroupProfileItem
from everos.memory.ports import Filter, GroupProfileIndex, Stored, all_of

from .dto import (
    GroupProfileEntry,
    ProfileSourceItem,
    SearchGroupProfile,
    SearchSource,
)

_PROFILE_SCAN_CAP = 2_000
"""Items one read folds. A group's profile grows by a few items per batch, so
this bounds a pathological history rather than a normal one."""


async def visible_profile_items(
    index: GroupProfileIndex, scope: ReadScope, *, where: Filter | None = None
) -> list[Stored[GroupProfileItem]]:
    """The current items of the scope's groups the asker may see."""
    if not scope.is_group:
        return []
    clause = scope.predicate if where is None else all_of(scope.predicate, where)
    return await index.find(clause, limit=_PROFILE_SCAN_CAP)


async def item_versions(
    index: GroupProfileIndex, scope: ReadScope
) -> list[Stored[GroupProfileItem]]:
    """Every version of every visible item, superseded ones too, newest first.

    A version whose successor lies outside the asker's windows still reads
    ``superseded``, but not by which entry: an id the asker cannot see is not
    handed out.
    """
    if not scope.is_group:
        return []
    found = await index.find(scope.history_predicate, limit=_PROFILE_SCAN_CAP)
    shown = {s.item.item_id for s in found}
    return [
        s
        if s.item.superseded_by is None or s.item.superseded_by in shown
        else Stored(
            ref=s.ref,
            scope=s.scope,
            item=s.item.model_copy(update={"superseded_by": None}),
        )
        for s in found
    ]


def visible_profile(
    stored: Sequence[Stored[GroupProfileItem]], *, cap: int
) -> tuple[SearchGroupProfile, GroupProfileView]:
    """The fold of ``stored``, each kind cut to ``cap``, shaped for a response.

    An item id (``gp_<YYYYMMDD>_<seq>``) is unique within its group only, so
    the items of a pool of groups are told apart by group and id: by id
    alone, two groups' first items of a day are one item.
    """
    by_id = {(s.item.owner_id, s.item.item_id): s for s in stored}
    view = current_profile([s.item for s in stored])

    def shaped(items: Sequence[GroupProfileItem]) -> list[GroupProfileEntry]:
        return [
            shape_profile_item(by_id[(i.owner_id, i.item_id)])
            for i in items[:cap]
            if (i.owner_id, i.item_id) in by_id
        ]

    return (
        SearchGroupProfile(
            topics=shaped(view.topics),
            decisions=shaped(view.decisions),
            dissent=shaped(view.dissent),
            roles=shaped(view.roles),
        ),
        view,
    )


def shape_profile_item(stored: Stored[GroupProfileItem]) -> GroupProfileEntry:
    """One item as the response carries it."""
    item = stored.item
    return GroupProfileEntry(
        id=stored.ref.id,
        item_id=item.item_id or stored.ref.entry_id,
        group_id=stored.scope.owner_id,
        kind=item.kind,
        status=item.status or item.own_status(),
        statement=item.statement,
        topic_id=item.topic_id,
        name=item.name,
        summary=item.summary,
        topic_status=item.topic_status,
        decided_by=list(item.decided_by),
        agreed_by=list(item.agreed_by),
        assignees=list(item.assignees),
        applies_to=list(item.applies_to),
        held_by=list(item.held_by),
        about=item.about,
        resolved_by=list(item.resolved_by),
        member_id=item.member_id,
        role=item.role,
        confidence=item.confidence,
        supersedes=item.supersedes,
        superseded_by=item.superseded_by,
        subject_ids=list(item.subject_ids),
        sender_ids=list(item.sender_ids),
        origin=item.origin,
        sources=[
            ProfileSourceItem(
                memcell_id=s.memcell_id,
                message_id=s.message_id,
                speaker_id=s.speaker_id,
                at=s.at,
                quote=s.quote,
            )
            for s in item.sources
        ],
        source=SearchSource(
            memcell_id=None, **{"from": item.source_start}, to=item.source_end
        ),
        observed_at=item.observed_at,
    )
