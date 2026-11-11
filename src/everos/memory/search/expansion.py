"""Relational expansion: the records a round's core links to (section 8.3).

A request often does not name the memory it needs ("draft the reminder"
needs the revised date). After each round of a group's multi-round search,
:class:`GroupLinks` follows the stored links of the round's new core -- never
the request's wording -- in this priority order:

1. **decision chain**: the decisions a core segment is evidence for, followed
   to their newest version, and that version's source segment's episode;
   then the open dissent about it;
2. **shared topic**: the topic's current decisions, by their source episodes;
3. **scene membership**: a core scene's episodes;
4. **subjects**: the most recent episodes about the same members (at most
   :data:`SUBJECT_MEMBERS` members, :data:`PER_SUBJECT` episodes each).

Every lookup carries the read scope's predicate (``where`` for episodes,
which also holds the request's filters; the scope's own predicate for
group-profile items, whose table has no session column), so a linked record
the asker may not see is never returned. A chain is followed only through
versions the asker can see. An episode's scene is not looked up: scene
membership is not indexed, so only a core scene leads to its episodes.
"""

from __future__ import annotations

__all__ = ["PER_SUBJECT", "SUBJECT_MEMBERS", "GroupLinks"]

from collections.abc import Sequence

from everos.memory.access import ReadScope
from everos.memory.models import Episode, GroupProfileItem, ImageMemory
from everos.memory.ports import (
    EpisodeIndex,
    Filter,
    GroupProfileIndex,
    Page,
    SceneStore,
    Stored,
    all_of,
    contains,
    eq,
    one_of,
)

SUBJECT_MEMBERS = 3
"""Members whose recent episodes one anchor leads to."""
PER_SUBJECT = 2
"""Recent episodes per member."""
_CHAIN_DEPTH = 20
"""Versions one chain is followed through at most (a bound, not a norm)."""


class GroupLinks:
    """A :class:`~everos.memory.search.llm_multiround.LinkExpander` for one read.

    Args:
        scope: The read scope (its predicate decides what a link may reach).
        where: The episode recall's ``where`` (scope predicate plus filters).
        episodes: The episode index (episodes and scenes).
        profile: The group-profile index.
        scenes: The scene store; ``None`` skips scene links.
    """

    def __init__(
        self,
        *,
        scope: ReadScope,
        where: Filter,
        episodes: EpisodeIndex,
        profile: GroupProfileIndex,
        scenes: SceneStore | None = None,
    ) -> None:
        self._scope = scope
        self._where = where
        self._episodes = episodes
        self._profile = profile
        self._scenes = scenes

    async def __call__(
        self, anchors: Sequence[Stored[Episode] | Stored[ImageMemory]]
    ) -> list[tuple[str, Stored[Episode]]]:
        """Linked episodes, highest priority first, each with its anchor's id."""
        out: list[tuple[str, Stored[Episode]]] = []
        seen = {a.ref.id for a in anchors}

        def take(anchor: str, found: Sequence[Stored[Episode]]) -> None:
            for record in found:
                if record.ref.id not in seen:
                    seen.add(record.ref.id)
                    out.append((anchor, record))

        for anchor in anchors:
            for found in await self._decision_links(anchor):
                take(anchor.ref.id, found)
        for anchor in anchors:
            take(anchor.ref.id, await self._scene_links(anchor))
        for anchor in anchors:
            take(anchor.ref.id, await self._subject_links(anchor))
        return out

    # ── links ─────────────────────────────────────────────────────────

    async def _decision_links(
        self, anchor: Stored[Episode] | Stored[ImageMemory]
    ) -> list[list[Stored[Episode]]]:
        """Chain heads' episodes, open dissent's, then the topics' decisions'."""
        memcell = _segment_of(anchor)
        if memcell is None:
            return []
        cited = await self._profile.find(
            all_of(
                self._scope.history_predicate,
                eq("kind", "decision"),
                contains("source_memcell_ids", memcell),
            )
        )
        heads = [h for h in [await self._head(d) for d in cited] if h is not None]
        head_ids = list(dict.fromkeys(h.item_id for h in heads if h.item_id))
        dissent = (
            await self._profile.find(
                all_of(
                    self._scope.predicate,
                    eq("kind", "dissent"),
                    one_of("about", head_ids),
                )
            )
            if head_ids
            else []
        )
        topics = list(dict.fromkeys(h.topic_id for h in heads if h.topic_id))
        siblings = (
            await self._profile.find(
                all_of(
                    self._scope.predicate,
                    eq("kind", "decision"),
                    one_of("topic_id", topics),
                )
            )
            if topics
            else []
        )
        open_dissent = [d.item for d in dissent if not d.item.resolved]
        return [
            await self._episodes_of(heads, exclude=memcell),
            await self._episodes_of(open_dissent, exclude=memcell),
            await self._episodes_of([s.item for s in siblings], exclude=memcell),
        ]

    async def _head(self, stored: Stored[GroupProfileItem]) -> GroupProfileItem | None:
        """The newest version of ``stored``'s chain the asker can see."""
        item = stored.item
        for _ in range(_CHAIN_DEPTH):
            if item.superseded_by is None:
                return item
            newer = await self._profile.find(
                all_of(
                    self._scope.history_predicate, eq("entry_id", item.superseded_by)
                ),
                limit=1,
            )
            if not newer:
                # The newer version lies outside the asker's windows: the chain
                # ends, for them, at a version they know is not the newest.
                return None
            item = newer[0].item
        return None

    async def _episodes_of(
        self, items: Sequence[GroupProfileItem], *, exclude: str
    ) -> list[Stored[Episode]]:
        """The episodes of the segments ``items`` rest on, newest first."""
        segments = [
            m
            for m in dict.fromkeys(m for i in items for m in i.source_memcell_ids)
            if m != exclude
        ]
        if not segments:
            return []
        hits = await self._episodes.find(
            all_of(
                self._where, eq("parent_type", "memcell"), one_of("parent_id", segments)
            )
        )
        return sorted(
            (h.stored for h in hits), key=lambda s: s.item.timestamp, reverse=True
        )

    async def _scene_links(
        self, anchor: Stored[Episode] | Stored[ImageMemory]
    ) -> list[Stored[Episode]]:
        item = anchor.item
        if (
            self._scenes is None
            or not isinstance(item, Episode)
            or item.parent_type != "scene"
        ):
            return []
        scene = await self._scenes.latest(anchor.scope, item.parent_id)
        if scene is None or not scene.item.member_ids:
            return []
        hits = await self._episodes.find(
            all_of(self._where, one_of("entry_id", list(scene.item.member_ids)))
        )
        return [h.stored for h in hits]

    async def _subject_links(
        self, anchor: Stored[Episode] | Stored[ImageMemory]
    ) -> list[Stored[Episode]]:
        found: list[Stored[Episode]] = []
        for member in list(anchor.item.subject_ids)[:SUBJECT_MEMBERS]:
            page = await self._episodes.paginate(
                all_of(
                    self._where,
                    eq("parent_type", "memcell"),
                    contains("subject_ids", member),
                ),
                Page(
                    sort_by="timestamp", number=1, size=PER_SUBJECT + 1, descending=True
                ),
            )
            found += [s for s in page.items if s.ref.id != anchor.ref.id][:PER_SUBJECT]
        return found


def _segment_of(anchor: Stored[Episode] | Stored[ImageMemory]) -> str | None:
    """The segment a core record came from; ``None`` for a scene or a merge."""
    item = anchor.item
    if item.parent_type != "memcell" or not item.parent_id:
        return None
    return item.parent_id
