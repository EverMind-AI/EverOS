"""How a group segment is shown to the extraction model, and whom it is about.

docs/group-memory.md, section 7.2. The extraction call is the one-to-one call;
only what it is given changes, and the change is code, not prompt text (D15):

* **Speaker labels** (:func:`label_cell`). Each message's speaker becomes the
  sender's roster ``display_name`` at the message's time, an assistant member
  ``"<display_name> (assistant)"``. The transcript format is unchanged: it
  already names the speaker of every message.
* **Roster block** (:func:`roster_block`). :data:`ROSTER_BLOCK`, registered
  as ``chat.group_roster_block``: who speaks in this part and who it names,
  with access roles and titles, and the rule that what the assistant says is
  its own reply, never something the group agreed. A text segment gets it in
  a ``<group_chat>`` block after the original episode request
  (:func:`.episode.build_original_prompt`); a picture segment after the
  operator's custom instructions (:func:`with_roster_block`).
* **Speaker with id** (:func:`speaker_with_id`). The group-profile and
  constraint-binding prompts (sections 7.5 and 8.4) show a labelled
  speaker's member id next to the name, ``Chen (u_chen)``: the id is what
  their replies must name people by.

After the call :func:`attribution` sets the record's ``sender_ids``,
``subject_ids`` and ``origin`` from the memcell and the roster -- never from
the model -- together with the segment's span (``source_start`` /
``source_end``), which every read compares with the asker's windows.
"""

from __future__ import annotations

__all__ = [
    "ROSTER_BLOCK",
    "Attribution",
    "attribution",
    "label_cell",
    "roster_block",
    "speaker_with_id",
    "with_roster_block",
]

from collections.abc import Sequence
from dataclasses import dataclass

from everalgo.types import ChatMessage

from everos.memory.groups import named_members, roster_names
from everos.memory.models import (
    Group,
    MemberKind,
    MembershipInterval,
    MemCell,
    Origin,
    Roster,
)
from everos.memory.prompts import PromptTemplate

ROSTER_BLOCK = PromptTemplate(
    id="chat.group_roster_block",
    version=1,
    parts=(
        "This conversation is a group chat{group}.",
        " People in this part: {people}.",
        ' "{name}" is the group\'s AI assistant.',
        " {names} are the group's AI assistants.",
        "Attribute every statement to the person who made it. What the assistant "
        "says is the assistant's reply or suggestion; it is something the group "
        "agreed only when a member says or confirms it.",
    ),
)
"""The roster block of section 7.2. Parts, in order: the opening sentence (the
group's quoted name, when it has one), the people of the segment, one
assistant, several assistants, the attribution rule."""

_ASSISTANT_SUFFIX = " (assistant)"


@dataclass(frozen=True)
class Attribution:
    """Who said a group record, whom it is about, whose messages it rests on.

    Attributes:
        sender_ids: Distinct senders of the segment's chat messages.
        subject_ids: Human members the text names; the human senders if none.
        origin: ``member`` / ``assistant`` / ``mixed``; ``None`` without a
            chat message.
        source_start: Epoch ms of the segment's first message.
        source_end: Epoch ms of its last message.
    """

    sender_ids: list[str]
    subject_ids: list[str]
    origin: Origin | None
    source_start: int
    source_end: int


def label_cell(cell: MemCell, roster: Roster) -> MemCell:
    """The cell with every speaker labelled by their roster name at that time."""
    items = [
        item.model_copy(update={"sender_name": _label(roster, item)})
        if getattr(item, "sender_id", None)
        else item
        for item in cell.items
    ]
    return cell.model_copy(update={"items": items})


def speaker_with_id(message: ChatMessage) -> str:
    """A labelled message's speaker with their member id: ``Chen (u_chen)``.

    The name is the label :func:`label_cell` gave (an assistant's ends in
    ``(assistant)``); a speaker whose name is their id shows the id alone.
    """
    name = message.sender_name or message.sender_id
    if not message.sender_id or name == message.sender_id:
        return name
    return f"{name} ({message.sender_id})"


def with_roster_block(custom_instructions: str | None, block: str) -> str:
    """The operator's custom instructions, then the roster block."""
    if custom_instructions and custom_instructions.strip():
        return f"{custom_instructions.rstrip()}\n\n{block}"
    return block


def roster_block(group: Group, roster: Roster, cell: MemCell) -> str:
    """The roster block for one segment: its speakers and the members it names.

    People are listed in the order they first speak, then the members the
    messages name who did not speak; assistants get their own sentence.
    """
    messages = _chat_messages(cell)
    end = _span(cell)[1]
    speakers = list(dict.fromkeys(m.sender_id for m in messages if m.sender_id))
    text = "\n".join(str(m.content or "") for m in messages)
    named = named_members(text, roster_names(roster, until=end))
    people: list[str] = []
    assistants: list[str] = []
    for member_id in dict.fromkeys([*speakers, *named]):
        at = next((m.timestamp for m in messages if m.sender_id == member_id), end)
        interval = _interval(roster, member_id, at)
        if _kind_of(interval, messages, member_id) == "assistant":
            assistants.append(f"{_name(interval, member_id)}{_ASSISTANT_SUFFIX}")
        else:
            people.append(_person(interval, member_id))
    intro, people_part, one, several, rule = ROSTER_BLOCK.parts
    first = intro.format(group=f' in "{group.name}"' if group.name else "")
    if people:
        first += people_part.format(people=", ".join(people))
    if len(assistants) == 1:
        first += one.format(name=assistants[0])
    elif assistants:
        quoted = [f'"{a}"' for a in assistants]
        first += several.format(names=", ".join(quoted[:-1]) + " and " + quoted[-1])
    return f"{first}\n{rule}"


def attribution(cell: MemCell, roster: Roster, *, texts: Sequence[str]) -> Attribution:
    """The attribution fields of the record written from ``cell``.

    ``texts`` are the record's own title and body: a human member (current or
    former at the segment's end) is a subject when one of their names occurs
    in them; when none does, the subjects are the human senders.
    """
    messages = _chat_messages(cell)
    start, end = _span(cell)
    senders = list(dict.fromkeys(m.sender_id for m in messages if m.sender_id))
    kinds = {
        m.sender_id: _kind_of(
            _interval(roster, m.sender_id, m.timestamp), [m], m.sender_id
        )
        for m in messages
    }
    humans = [s for s in senders if kinds.get(s) == "human"]
    named = named_members(
        " ".join(texts), roster_names(roster, until=end, kind="human")
    )
    return Attribution(
        sender_ids=senders,
        subject_ids=named or humans,
        origin=_origin(set(kinds.values())),
        source_start=start,
        source_end=end,
    )


def _origin(kinds: set[MemberKind]) -> Origin | None:
    if not kinds:
        return None
    if kinds == {"human"}:
        return "member"
    if kinds == {"assistant"}:
        return "assistant"
    return "mixed"


def _chat_messages(cell: MemCell) -> list[ChatMessage]:
    return [item for item in cell.items if isinstance(item, ChatMessage)]


def _span(cell: MemCell) -> tuple[int, int]:
    times = [t for t in (getattr(i, "timestamp", 0) for i in cell.items) if t]
    if not times:
        return cell.timestamp, cell.timestamp
    return min(times), max(times)


def _interval(roster: Roster, member_id: str, at: int) -> MembershipInterval | None:
    """The member's interval at ``at``; else their latest one begun by then."""
    current = roster.interval_at(member_id, at)
    if current is not None:
        return current
    begun = [i for i in roster.intervals_of(member_id) if i.joined_at <= at]
    if begun:
        return max(begun, key=lambda i: i.joined_at)
    later = roster.intervals_of(member_id)
    return min(later, key=lambda i: i.joined_at) if later else None


def _kind_of(
    interval: MembershipInterval | None,
    messages: Sequence[ChatMessage],
    member_id: str,
) -> MemberKind:
    """The roster's kind; without an interval, the role the messages carry."""
    if interval is not None:
        return interval.kind
    roles = {m.role for m in messages if m.sender_id == member_id}
    return "assistant" if roles == {"assistant"} else "human"


def _name(interval: MembershipInterval | None, member_id: str) -> str:
    return interval.display_name if interval is not None else member_id


def _person(interval: MembershipInterval | None, member_id: str) -> str:
    if interval is None:
        return member_id
    details = ", ".join(p for p in (interval.role, interval.title) if p)
    return f"{interval.display_name} ({details})"


def _label(roster: Roster, item: object) -> str:
    sender = str(getattr(item, "sender_id", ""))
    at = int(getattr(item, "timestamp", 0) or 0)
    interval = _interval(roster, sender, at)
    if interval is None:
        fallback = getattr(item, "sender_name", None) or sender
        if getattr(item, "role", None) == "assistant":
            return f"{fallback}{_ASSISTANT_SUFFIX}"
        return str(fallback)
    if interval.kind == "assistant":
        return f"{interval.display_name}{_ASSISTANT_SUFFIX}"
    return interval.display_name
