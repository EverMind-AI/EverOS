"""The constraint block's rules layer (docs/group-memory.md, section 8.4).

Built once the search loop has stopped, from its core:

1. **Linked items.** The group-profile decisions, dissent and topics whose
   evidence lies in a core segment become constraints directly: their quotes,
   speakers and statuses were verified when they were written. A decision is
   followed to the newest version the asker can see, which becomes the
   constraint (``current``, or ``disputed`` while a dissent about it is open)
   with the versions it replaced listed in ``supersedes``; a decision whose
   newer version the asker cannot see stays as it is, ``superseded``. The
   open dissent about a constrained decision is a constraint of its own.
2. **Metadata.** A core episode or picture memory no linked item rests on gets
   a constraint of its own -- its subjects, speakers, segment and time -- whose
   quote is the segment's message sharing the most words with the record.

Every quote is checked once more against the message it cites, in the
segment's stored cell read without its forgotten messages: a constraint whose
quote is no longer a verbatim part of a readable message (its segment gone,
or the message forgotten) is left out. Every lookup carries
the read scope's predicate, so a constraint never rests on an item the asker
cannot see.
"""

from __future__ import annotations

__all__ = ["QUOTE_CHARS", "build_constraints"]

import re
from collections.abc import Sequence

from everalgo.types import ChatMessage, MemCell

from everos.memory.access import ReadScope
from everos.memory.groups import readable_segments
from everos.memory.models import GroupProfileItem
from everos.memory.ports import (
    GroupProfileIndex,
    MemcellStore,
    MessageForgetStore,
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


async def build_constraints(
    *,
    scope: ReadScope,
    core: Sequence[CoreRecord],
    profile: GroupProfileIndex,
    memcells: MemcellStore,
    forgotten: MessageForgetStore,
) -> tuple[list[ConstraintItem], dict[str, MemCell]]:
    """The rules-layer constraints of one group search, in block order.

    Also returns the segments read on the way, by memcell id: what the model
    layer is shown and checks its quotes against.
    """
    if not scope.is_group or not core:
        return [], {}
    segments = list(dict.fromkeys(m for m in (_segment(r) for r in core) if m))
    linked = (
        await profile.find(
            all_of(
                scope.history_predicate,
                one_of("kind", list(_KINDS)),
                any_of(*(contains("source_memcell_ids", m) for m in segments)),
            )
        )
        if segments
        else []
    )
    built: list[tuple[GroupProfileItem, str, list[GroupProfileItem]]] = []
    seen: set[str] = set()
    for stored in sorted(linked, key=lambda s: _KINDS.index(s.item.kind)):
        item, chain = await _newest(profile, scope, stored.item)
        if item.item_id in seen:
            continue
        seen.add(item.item_id or "")
        built.append((item, stored.scope.owner_id, chain))
    decisions = [i.item_id for i, _, _ in built if i.kind == "decision" and i.item_id]
    dissent = (
        await profile.find(
            all_of(scope.predicate, eq("kind", "dissent"), one_of("about", decisions))
        )
        if decisions
        else []
    )
    disputed = {d.item.about for d in dissent if not d.item.resolved}
    for stored in dissent:
        if stored.item.item_id not in seen and not stored.item.resolved:
            seen.add(stored.item.item_id or "")
            built.append((stored.item, stored.scope.owner_id, []))

    # A forgotten message is quoted nowhere: the segments are read without
    # the messages a live forget names (section 9.4).
    cells = await readable_segments(
        forgotten,
        await memcells.cells_by_id(
            list(
                dict.fromkeys(
                    [*segments, *(s.memcell_id for i, _, _ in built for s in i.sources)]
                )
            )
        ),
    )
    out: list[ConstraintItem] = []
    covered: set[str] = set()
    for item, group_id, chain in built:
        constraint = _from_item(
            item, group_id, chain, disputed=disputed, cells=cells, scope=scope
        )
        if constraint is not None:
            out.append(constraint)
            covered.update(item.source_memcell_ids)
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
    profile: GroupProfileIndex, scope: ReadScope, item: GroupProfileItem
) -> tuple[GroupProfileItem, list[GroupProfileItem]]:
    """The newest visible version of ``item``'s chain, and the versions it replaced.

    The chain is followed forward through ``superseded_by`` while the next
    version is visible to the asker; when it is not, the chain stops there
    and the item comes back as it is, marked ``superseded``. The replaced
    versions are then walked back through ``supersedes``, newest first, as
    far as the asker can see.
    """
    current = item
    for _ in range(_CHAIN_DEPTH):
        if current.superseded_by is None:
            break
        newer = await _version(profile, scope, current.superseded_by)
        if newer is None:
            break
        current = newer
    replaced: list[GroupProfileItem] = []
    older_id = current.supersedes
    for _ in range(_CHAIN_DEPTH):
        if older_id is None:
            break
        older = await _version(profile, scope, older_id)
        if older is None:
            break
        replaced.append(older)
        older_id = older.supersedes
    return current, replaced


async def _version(
    profile: GroupProfileIndex, scope: ReadScope, entry_id: str
) -> GroupProfileItem | None:
    """One version of a chain, if the asker may see it (forgotten ones never)."""
    found = await profile.find(
        all_of(scope.history_predicate, eq("entry_id", entry_id)), limit=1
    )
    return found[0].item if found else None


def _from_item(
    item: GroupProfileItem,
    group_id: str,
    chain: Sequence[GroupProfileItem],
    *,
    disputed: set[str | None],
    cells: dict[str, MemCell],
    scope: ReadScope,
) -> ConstraintItem | None:
    source = next(
        (
            s
            for s in item.sources
            if _verbatim(cells.get(s.memcell_id), s.message_id, s.quote)
        ),
        None,
    )
    if source is None:
        return None
    status = item.status or item.own_status()
    if item.kind == "decision" and status == "current" and item.item_id in disputed:
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
            quote=source.quote,
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
            quote=_normalise(str(message.content or ""))[:QUOTE_CHARS],
        ),
        audience=audience_of(scope, record.group_id, start=start, end=end),
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


def _verbatim(cell: MemCell | None, message_id: str, quote: str) -> bool:
    """Whether ``quote`` is still a verbatim part of the message it cites."""
    if cell is None or not quote:
        return False
    for message in cell.items:
        if isinstance(message, ChatMessage) and message.id == message_id:
            return quote in _normalise(str(message.content or ""))
    return False


def _statement(item: GroupProfileItem) -> str:
    if item.kind == "topic":
        return f"{item.name}: {item.summary}" if item.summary else item.name
    return item.statement


def _segment(record: CoreRecord) -> str | None:
    """The segment a core record came from; ``None`` for a scene."""
    return record.source.memcell_id if record.source is not None else None


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _normalise(text: str) -> str:
    return " ".join(text.split())
