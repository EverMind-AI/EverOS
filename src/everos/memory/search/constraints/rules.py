"""The constraint block's rules layer (docs/group-memory.md, section 8.4).

Built once the search loop has stopped, from its core:

1. **Linked items** (:func:`link_items`). The group-profile decisions,
   dissent and topics whose evidence lies in a core segment become
   constraints directly: their quotes, speakers and statuses were verified
   when they were written. A decision is followed to the newest version the
   asker can see, which becomes the constraint (``current``, or ``disputed``
   while a dissent about it is open) with the versions it replaced listed in
   ``supersedes``; a decision whose newer version the asker cannot see stays
   as it is, ``superseded``. The open dissent about a constrained decision is
   a constraint of its own. The same items are what a group search attaches
   as its ``group_profile`` by default (section 5.3). An item id is unique
   within its group only, so every lookup and every comparison names the
   group too.
2. **Metadata.** A core episode or picture memory no linked item rests on gets
   a constraint of its own -- its subjects, speakers, segment and time -- whose
   quote is the segment's message sharing the most words with the record.

Every quote is checked once more against the message it cites, in the
segment's stored cell read without its forgotten messages and without the
references to the pictures its group forgot: a constraint whose quote is no
longer a verbatim part of a readable message (its segment gone, or the
message forgotten) is left out, and no quote names a forgotten picture. The
check is the write path's (:func:`everos.memory.groups.verbatim_part`), and
the quote returned is the message's own words. Every lookup carries the read
scope's predicate, so a constraint never rests on an item the asker cannot
see.
"""

from __future__ import annotations

__all__ = [
    "QUOTE_CHARS",
    "CoreRecord",
    "LinkedItem",
    "LinkedItems",
    "build_constraints",
    "link_items",
]

import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from everalgo.types import ChatMessage, MemCell

from everos.memory.access import ReadScope
from everos.memory.extract.pipeline.pictures import image_ids_in, without_pictures
from everos.memory.groups import normalise_space, readable_segments, verbatim_part
from everos.memory.models import GroupProfileItem, ProfileSource
from everos.memory.ports import (
    GroupProfileIndex,
    MemcellStore,
    MessageForgetStore,
    Stored,
    all_of,
    any_of,
    contains,
    eq,
    one_of,
)

from ..audience import audience_of
from ..dto import (
    ConstraintItem,
    ConstraintScopeItem,
    ConstraintSourceItem,
    SearchEpisodeItem,
    SearchImageItem,
    SupersededStatement,
)

QUOTE_CHARS = 300
"""A metadata constraint quotes at most this many characters of its message
(a prefix, so the quote stays a verbatim part of it)."""

_KINDS = ("decision", "dissent", "topic")
_CHAIN_DEPTH = 20
"""Versions one chain is followed through at most (a bound, not a norm)."""
_WORD = re.compile(r"\w+", re.UNICODE)


type CoreRecord = SearchEpisodeItem | SearchImageItem
"""A record of the core, as the search returns it."""


@dataclass(frozen=True)
class LinkedItem:
    """A group-profile item linked to the core, as the asker can see it.

    ``stored`` is the newest visible version of the chain whose evidence lies
    in a core segment (or an open dissent about such a decision); ``chain``
    the versions it replaced, newest first, as far as the asker can see.
    """

    stored: Stored[GroupProfileItem]
    chain: tuple[GroupProfileItem, ...] = ()

    @property
    def item(self) -> GroupProfileItem:
        return self.stored.item

    @property
    def group_id(self) -> str:
        return self.stored.scope.owner_id


@dataclass(frozen=True)
class LinkedItems:
    """Step 1 of the rules layer: the items linked to one search's core."""

    items: tuple[LinkedItem, ...] = ()
    """In block order: decisions, dissent, topics, then the open dissent about
    the decisions."""
    disputed: frozenset[tuple[str, str]] = frozenset()
    """``(group id, decision id)`` of every linked decision an open dissent is
    about."""


async def link_items(
    *, scope: ReadScope, core: Sequence[CoreRecord], profile: GroupProfileIndex
) -> LinkedItems:
    """The group-profile items linked to ``core`` (step 1), in block order.

    Every read carries the read scope's predicate: an item the asker cannot
    see is never linked, and a chain stops at the last version they can see.
    """
    if not scope.is_group or not core:
        return LinkedItems()
    segments = list(dict.fromkeys(m for m in (_segment(r) for r in core) if m))
    if not segments:
        return LinkedItems()
    linked = await profile.find(
        all_of(
            scope.history_predicate,
            one_of("kind", list(_KINDS)),
            any_of(*(contains("source_memcell_ids", m) for m in segments)),
        )
    )
    built: list[LinkedItem] = []
    seen: set[tuple[str, str]] = set()
    for stored in sorted(linked, key=lambda s: _KINDS.index(s.item.kind)):
        newest, chain = await _newest(profile, scope, stored)
        key = _key(newest)
        # An update may rewrite an item into another kind: a chain whose
        # newest visible version is a role is no longer a constraint.
        if key in seen or newest.item.kind not in _KINDS:
            continue
        seen.add(key)
        built.append(LinkedItem(stored=newest, chain=tuple(chain)))
    decisions: dict[str, list[str]] = {}
    for linked_item in built:
        if linked_item.item.kind == "decision" and linked_item.item.item_id:
            decisions.setdefault(linked_item.group_id, []).append(
                linked_item.item.item_id
            )
    dissent = (
        await profile.find(
            all_of(
                scope.predicate,
                eq("kind", "dissent"),
                any_of(
                    *(
                        all_of(eq("owner_id", group_id), one_of("about", ids))
                        for group_id, ids in decisions.items()
                    )
                ),
            )
        )
        if decisions
        else []
    )
    disputed = frozenset(
        (d.scope.owner_id, d.item.about)
        for d in dissent
        if d.item.about and not d.item.resolved
    )
    for stored in dissent:
        if _key(stored) not in seen and not stored.item.resolved:
            seen.add(_key(stored))
            built.append(LinkedItem(stored=stored))
    return LinkedItems(items=tuple(built), disputed=disputed)


async def build_constraints(
    *,
    scope: ReadScope,
    core: Sequence[CoreRecord],
    linked: LinkedItems,
    memcells: MemcellStore,
    forgotten: MessageForgetStore,
    forgotten_pictures: Callable[[], Awaitable[Mapping[str, Collection[str]]]]
    | None = None,
) -> tuple[list[ConstraintItem], dict[str, MemCell]]:
    """The rules-layer constraints of one group search, in block order.

    ``linked`` is :func:`link_items` over the same core. Also returns the
    segments read on the way, by memcell id: what the model layer is shown
    and checks its quotes against. ``forgotten_pictures`` reads the pictures
    each group forgot (by group id), and only when a segment read shows a
    picture: a segment is read without their references, so no quote names
    a forgotten picture.
    """
    if not scope.is_group or not core:
        return [], {}
    segments = list(dict.fromkeys(m for m in (_segment(r) for r in core) if m))

    # A forgotten message is quoted nowhere: the segments are read without
    # the messages a live forget names (section 9.4), and without the
    # references to the pictures their group forgot.
    # A segment is one session's, and a session is one group's.
    groups_of = {
        s.memcell_id: li.group_id for li in linked.items for s in li.item.sources
    }
    for record in core:
        segment = _segment(record)
        if segment is not None and record.group_id is not None:
            groups_of[segment] = record.group_id
    cells = await readable_segments(
        forgotten,
        await memcells.cells_by_id(
            list(
                dict.fromkeys(
                    [
                        *segments,
                        *(s.memcell_id for li in linked.items for s in li.item.sources),
                    ]
                )
            )
        ),
    )
    if forgotten_pictures is not None and any(map(_shows_a_picture, cells.values())):
        pictures = await forgotten_pictures()
        cells = {
            memcell_id: without_pictures(
                cell, pictures.get(groups_of.get(memcell_id, ""), ())
            )
            for memcell_id, cell in cells.items()
        }
    out: list[ConstraintItem] = []
    covered: set[str] = set()
    for li in linked.items:
        constraint = _from_item(
            li.item,
            li.group_id,
            li.chain,
            disputed=linked.disputed,
            cells=cells,
            scope=scope,
        )
        if constraint is not None:
            out.append(constraint)
            covered.update(li.item.source_memcell_ids)
    for record in core:
        segment = _segment(record)
        if segment is None or segment in covered:
            continue
        constraint = _from_record(record, cells.get(segment), segment, scope)
        if constraint is not None:
            out.append(constraint)
    numbered = [c.model_copy(update={"id": f"c{n}"}) for n, c in enumerate(out, 1)]
    return numbered, cells


async def _newest(
    profile: GroupProfileIndex, scope: ReadScope, stored: Stored[GroupProfileItem]
) -> tuple[Stored[GroupProfileItem], list[GroupProfileItem]]:
    """The newest visible version of ``stored``'s chain, and the versions it replaced.

    The chain is followed forward through ``superseded_by`` while the next
    version is visible to the asker; when it is not, the chain stops there
    and the item comes back as it is, marked ``superseded``. The replaced
    versions are then walked back through ``supersedes``, newest first, as
    far as the asker can see. Every version is looked up in the item's own
    group.
    """
    group_id = stored.scope.owner_id
    current = stored
    for _ in range(_CHAIN_DEPTH):
        if current.item.superseded_by is None:
            break
        newer = await _version(profile, scope, group_id, current.item.superseded_by)
        if newer is None:
            break
        current = newer
    replaced: list[GroupProfileItem] = []
    older_id = current.item.supersedes
    for _ in range(_CHAIN_DEPTH):
        if older_id is None:
            break
        older = await _version(profile, scope, group_id, older_id)
        if older is None:
            break
        replaced.append(older.item)
        older_id = older.item.supersedes
    return current, replaced


async def _version(
    profile: GroupProfileIndex, scope: ReadScope, group_id: str, entry_id: str
) -> Stored[GroupProfileItem] | None:
    """One version of a chain in ``group_id``, if the asker may see it.

    Forgotten versions never.
    """
    found = await profile.find(
        all_of(
            scope.history_predicate,
            eq("owner_id", group_id),
            eq("entry_id", entry_id),
        ),
        limit=1,
    )
    return found[0] if found else None


def _from_item(
    item: GroupProfileItem,
    group_id: str,
    chain: Sequence[GroupProfileItem],
    *,
    disputed: frozenset[tuple[str, str]],
    cells: dict[str, MemCell],
    scope: ReadScope,
) -> ConstraintItem | None:
    cited = _cited(item, cells)
    if cited is None:
        return None
    source, quote = cited
    status = item.status or item.own_status()
    if (
        item.kind == "decision"
        and status == "current"
        and (group_id, item.item_id or "") in disputed
    ):
        status = "disputed"
    return ConstraintItem(
        id="",
        statement=_statement(item),
        kind=item.kind,  # type: ignore[arg-type]
        record_id=item.item_id or "",
        subject_ids=list(item.subject_ids),
        scope=ConstraintScopeItem(
            group_id=group_id, applies_to=list(item.applies_to) or "group"
        ),
        status=status,
        supersedes=[
            SupersededStatement(statement=_statement(old), at=old.source_end)
            for old in chain
        ],
        decided_by=list(item.decided_by),
        agreed_by=list(item.agreed_by),
        held_by=list(item.held_by),
        origin=item.origin,
        source=ConstraintSourceItem(
            memcell_id=source.memcell_id,
            message_id=source.message_id,
            speaker_id=source.speaker_id,
            at=source.at,
            quote=quote,
        ),
        audience=audience_of(
            scope, group_id, start=item.source_start, end=item.source_end
        ),
    )


def _from_record(
    record: CoreRecord, cell: MemCell | None, segment: str, scope: ReadScope
) -> ConstraintItem | None:
    """A metadata constraint: the record, quoting its best-matching message."""
    if cell is None or record.group_id is None or record.source is None:
        return None
    if isinstance(record, SearchImageItem):
        kind, statement = "picture", record.description
        text = f"{record.description} {record.ocr}"
    else:
        kind, statement = "episode", record.summary or record.subject
        text = f"{record.subject} {record.summary} {record.episode}"
    message = _best_message(cell, text)
    if message is None:
        return None
    start, end = record.source.from_, record.source.to
    return ConstraintItem(
        id="",
        statement=statement,
        kind=kind,  # type: ignore[arg-type]
        record_id=record.id,
        subject_ids=list(record.subject_ids),
        scope=ConstraintScopeItem(group_id=record.group_id),
        status="current",
        origin=record.origin,
        source=ConstraintSourceItem(
            memcell_id=segment,
            message_id=message.id,
            speaker_id=message.sender_id,
            at=message.timestamp,
            quote=normalise_space(str(message.content or ""))[:QUOTE_CHARS],
        ),
        audience=audience_of(scope, record.group_id, start=start, end=end),
    )


def _shows_a_picture(cell: MemCell) -> bool:
    """Whether one of the cell's messages refers to a picture."""
    return any(
        image_ids_in(str(getattr(item, "content", "") or "")) for item in cell.items
    )


def _best_message(cell: MemCell, text: str) -> ChatMessage | None:
    """The cell's message sharing the most words with ``text``; the first on a tie."""
    wanted = set(_words(text))
    best: ChatMessage | None = None
    best_score = -1
    for message in cell.items:
        if (
            not isinstance(message, ChatMessage)
            or not str(message.content or "").strip()
        ):
            continue
        score = len(wanted & set(_words(str(message.content))))
        if score > best_score:
            best, best_score = message, score
    return best


def _cited(
    item: GroupProfileItem, cells: dict[str, MemCell]
) -> tuple[ProfileSource, str] | None:
    """The item's first source still quoted verbatim, with the message's words."""
    for source in item.sources:
        quote = _verbatim(cells.get(source.memcell_id), source.message_id, source.quote)
        if quote is not None:
            return source, quote
    return None


def _verbatim(cell: MemCell | None, message_id: str, quote: str) -> str | None:
    """The part of the cited message ``quote`` still copies, if it does."""
    if cell is None:
        return None
    for message in cell.items:
        if isinstance(message, ChatMessage) and message.id == message_id:
            return verbatim_part(quote, str(message.content or ""))
    return None


def _statement(item: GroupProfileItem) -> str:
    if item.kind == "topic":
        return f"{item.name}: {item.summary}" if item.summary else item.name
    return item.statement


def _key(stored: Stored[GroupProfileItem]) -> tuple[str, str]:
    """An item's identity across a pool of groups: its group and its id."""
    return stored.scope.owner_id, stored.item.item_id or ""


def _segment(record: CoreRecord) -> str | None:
    """The segment a core record came from; ``None`` for a scene."""
    return record.source.memcell_id if record.source is not None else None


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]
