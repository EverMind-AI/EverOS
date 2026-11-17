"""The group-profile update prompt, ``chat.group_profile_update`` (section 7.5).

One call per batch of segments. It is shown:

* who is who: every member of the roster by id, with display name, access
  role and title, the assistants marked, and the roster block's attribution
  rule (an assistant's words are its suggestion, never a group decision);
* the current profile, every item with its id: the active topics (at most
  :data:`MAX_TOPICS`, most recently active first -- replaced topics stay
  stored and are not shown), each summary cut to its first
  :data:`TOPIC_SUMMARY_SHOWN` characters, the current decisions, open dissent
  and active roles;
* the new segments, each opened by ``=== SEGMENT <memcell_id> ===``, one line
  per message: ``[<message_id>] <time> <speaker> (<member_id>): <text>`` --
  the id is what the reply names people by (version 2; version 1 showed the
  name alone).

It answers with operations (:mod:`.reply`), each citing evidence as
``(segment, message, quote)``. Code checks every one (:mod:`.verify`). A
``topic.new`` gives its topic a label of the model's own (``new_1``, ...) in
``topic_id``, by which the reply's other operations name the new topic in
``topic_id`` or ``about``; code mints the topic's id (version 3; version 2 had
no way to name a topic created in the same reply).

Version 4 (2026-11-17) makes the call smaller and the replacements explicit.
The reply shows each operation in a shape of its own fields and leaves out a
field without a value, where version 3 had one shape with every field, and a
topic's summary is at most :data:`TOPIC_SUMMARY_MAX` characters; the profile
shows each summary cut to :data:`TOPIC_SUMMARY_SHOWN`. A ``decision.new``
always says what it ``replaces``: a listed decision's id, which code turns
into the supersede it checks (:mod:`.verify`), or ``none``. Version 3 asked
for a separate ``decision.supersede``, which the model almost never wrote
(1 of 197 operations on a GroupAssistBench slice, 0 of 940 on EverMemBench,
2026-11-16) while recording the new value as another current decision.

The prompt is zero-shot and names the conversation's language, resolved in
code from the segments' human messages; only a batch without a letter in it
leaves the language to the model. The text is registered as
:data:`GROUP_PROFILE_UPDATE` (id, version, fingerprint pinned by tests); a
change to it bumps the version.
"""

from __future__ import annotations

__all__ = [
    "GROUP_PROFILE_UPDATE",
    "MAX_TOPICS",
    "TOPIC_SUMMARY_MAX",
    "TOPIC_SUMMARY_SHOWN",
    "Segment",
    "build_messages",
    "render_members",
    "render_profile",
    "render_segments",
]

from collections.abc import Sequence
from dataclasses import dataclass

from everalgo.llm import LLMMessage
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)

from everos.core.utils.datetime import from_timestamp_ms
from everos.memory.extract.pipeline.group_render import label_cell, speaker_with_id
from everos.memory.models import GroupProfileItem, MembershipInterval, Roster
from everos.memory.prompts import PromptTemplate

from .fold import GroupProfileView

MAX_TOPICS = 20
"""Active topics shown to the model, most recently active first."""
TOPIC_SUMMARY_MAX = 300
"""Characters a topic's summary may have; the prompt asks for no more."""
TOPIC_SUMMARY_SHOWN = 200
"""Characters of a listed topic's summary the prompt shows; a longer one is
cut and ends with an ellipsis."""
MAX_DECISIONS = 40
MAX_DISSENT = 20
MAX_ROLES = 30

GROUP_PROFILE_UPDATE = PromptTemplate(
    id="chat.group_profile_update",
    version=4,
    parts=(
        # 0: the system prompt; {members}, {language_rule}, {summary_max} and
        # {summary_shown} are filled in.
        """You keep the shared profile of a group chat: the topics the group \
discusses, the decisions its members make, the disagreements that are still \
open, and the roles members play. You read new parts of the conversation and \
reply with the operations that bring the profile up to date. You never edit \
the profile: every operation adds a record, and an operation that changes an \
existing record names the record it replaces.

<members>
{members}
Attribute every statement to the person who made it. What the assistant says \
is the assistant's reply or suggestion; it is something the group agreed only \
when a member says or confirms it.
</members>

<operations>
- topic.new: a substantive thread the profile does not have yet. Give name, \
summary and status, and topic_id = a label for the new topic (new_1, new_2, \
...) that the other operations of this reply use as topic_id or about to name \
it.
- topic.update: a listed topic whose summary or status changed. target = its \
topic id (tp_...). Give name, summary and status as they are now.
- decision.new: something the members decided or agreed to do. Give \
statement, replaces, decided_by (who made the decision), agreed_by (who \
agreed), assignees (who will carry it out), applies_to (the members it \
applies to; left out when it applies to the whole group) and topic_id (a \
listed topic id or a new topic's label). replaces = the id of the listed \
decision this one changes or overturns -- a new value for the same thing: a \
new date, price, place, choice or owner, or a final result that replaces an \
interim one -- or "none" when it adds something: another decision, a next \
step, or progress on a listed decision. A replaced decision is no longer \
current.
- dissent.open: a member disagrees with a decision or a topic's direction and \
has not been brought round. Give statement (what they object to and why), \
held_by and about (a listed decision id or topic id, or a new topic's label).
- dissent.resolve: a listed open dissent is settled. target = its id; \
statement = how it was settled.
- role.add: a member clearly plays one of the roles below. Give member_id, \
role and confidence (strong when several messages show it, weak when one \
does).
- role.retire: a listed role the member no longer plays. target = its id.
</operations>

<roles>
- decision_maker: makes or finalises the group's decisions.
- opinion_leader: others follow or defer to their views.
- topic_initiator: starts the threads the group then discusses.
- execution_promoter: pushes tasks forward, follows up and chases progress.
- core_contributor: supplies the substance: facts, plans, work.
- coordinator: arranges people, times and logistics.
- info_summarizer: sums up discussions and outcomes for the others.
</roles>

<rules>
1. Evidence. Every operation cites one or more messages as {{"segment", \
"message", "quote"}}: segment = the id after "=== SEGMENT", message = the id \
in square brackets at the start of the message's line, quote = words copied \
exactly from that message's text -- not its time or speaker. Copy; never \
paraphrase, shorten inside, correct or translate a quote.
2. People. Name people by their member ids -- the ids in <members>, also \
shown in parentheses after each speaker's name -- in decided_by, agreed_by, \
assignees, applies_to, held_by and member_id. Everyone you name in \
decided_by, agreed_by or held_by, and the member_id of a role, must have \
written a message the operation cites: cite one message of each of them in \
evidence, or leave them out.
3. The assistant. A decision needs at least one member's own message that \
makes or confirms it; the assistant's suggestion alone is not a decision. \
Never give the assistant a role or a dissent.
4. Topics. A topic is a substantive thread: a plan, a problem, a piece of \
work, a question the group works through. Scheduling chatter, greetings, \
thanks, bot notices and one-word confirmations are not topics. Status: \
exploring (being discussed), disagreement (members disagree), consensus \
(members agree), implemented (done). A summary says what the thread is about \
and where it stands now, in at most {summary_max} characters. <profile> shows \
each listed topic's summary cut to its first {summary_shown} characters (… \
marks the cut); a topic.update writes the whole summary anew.
5. Ids. Use only ids that appear in <profile>, and the labels of the topics \
this reply creates. Never invent an id.
6. Only what changed. Do not repeat what the profile already says. If \
nothing changed, reply {{"operations": []}}.
7. {language_rule}
</rules>

Reply with JSON only: {{"operations": [...]}}, each operation in its shape \
below. Write only the fields of its shape, and leave out a field that has no \
value (an empty text or list); a decision.new always gives replaces. Every \
evidence entry is {{"segment": "", "message": "", "quote": ""}}.
{{"op": "topic.new", "topic_id": "new_1", "name": "", "summary": "", \
"status": "", "evidence": []}}
{{"op": "topic.update", "target": "", "name": "", "summary": "", "status": \
"", "evidence": []}}
{{"op": "decision.new", "statement": "", "replaces": "none", "decided_by": \
[], "agreed_by": [], "assignees": [], "applies_to": [], "topic_id": "", \
"evidence": []}}
{{"op": "dissent.open", "statement": "", "held_by": [], "about": "", \
"evidence": []}}
{{"op": "dissent.resolve", "target": "", "statement": "", "evidence": []}}
{{"op": "role.add", "member_id": "", "role": "", "confidence": "", \
"evidence": []}}
{{"op": "role.retire", "target": "", "evidence": []}}""",
        # 1: the user message.
        """<profile>
{profile}
</profile>

<segments>
{segments}
</segments>""",
        # 2: a member line; 3: an assistant member line.
        "- {member_id}: {name}{details}",
        "- {member_id}: {name} (assistant) -- the group's AI assistant",
    ),
)
"""The update prompt: system text, user text, and the two member-line shapes."""


@dataclass(frozen=True)
class Segment:
    """One segment of the batch: its memcell id and its cell."""

    memcell_id: str
    cell: MemCell


def build_messages(
    *,
    roster: Roster,
    profile: GroupProfileView,
    segments: Sequence[Segment],
    language: OutputLanguage | None,
) -> list[LLMMessage]:
    """The two messages of one update call."""
    system, user, _, _ = GROUP_PROFILE_UPDATE.parts
    rule = build_language_rule(language, fallback=PARTICIPANT_LANGUAGE_RULE)
    return [
        LLMMessage(
            role="system",
            content=system.format(
                members=render_members(roster, segments),
                language_rule=rule,
                summary_max=TOPIC_SUMMARY_MAX,
                summary_shown=TOPIC_SUMMARY_SHOWN,
            ),
        ),
        LLMMessage(
            role="user",
            content=user.format(
                profile=render_profile(profile),
                segments=render_segments(segments, roster),
            ),
        ),
    ]


def render_members(roster: Roster, segments: Sequence[Segment]) -> str:
    """Every member of the roster by id, as they were at the batch's end.

    Former members are listed too: their messages may be in the batch. A
    member's latest interval begun by then describes them; one who joined
    later is described by their first.
    """
    _, _, person, assistant = GROUP_PROFILE_UPDATE.parts
    end = max((s.cell.timestamp for s in segments), default=0)
    latest: dict[str, MembershipInterval] = {}
    for interval in sorted(roster.intervals, key=lambda i: i.joined_at):
        if interval.member_id not in latest or interval.joined_at <= end:
            latest[interval.member_id] = interval
    lines: list[str] = []
    for member_id, interval in latest.items():
        if interval.kind == "assistant":
            lines.append(
                assistant.format(member_id=member_id, name=interval.display_name)
            )
            continue
        details = ", ".join(p for p in (interval.role, interval.title) if p)
        lines.append(
            person.format(
                member_id=member_id,
                name=interval.display_name,
                details=f" ({details})" if details else "",
            )
        )
    return "\n".join(lines) or "(no roster)"


def render_profile(profile: GroupProfileView) -> str:
    """The current items with their ids, by kind; ``(empty)`` sections kept.

    A topic's summary is cut to its first :data:`TOPIC_SUMMARY_SHOWN`
    characters.
    """
    topics = [
        f"- [{t.topic_id}] {t.name} ({t.topic_status or 'exploring'}): "
        f"{_shown_summary(t.summary)}"
        for t in profile.topics[:MAX_TOPICS]
    ]
    shown = {t.topic_id for t in profile.topics[:MAX_TOPICS]}
    decisions = [
        _decision_line(d)
        for d in profile.decisions
        if d.topic_id is None or d.topic_id in shown
    ][:MAX_DECISIONS]
    dissent = [
        f"- [{d.item_id}] held by {', '.join(d.held_by) or '?'}"
        f"{f' about {d.about}' if d.about else ''}: {d.statement}"
        for d in profile.dissent[:MAX_DISSENT]
    ]
    roles = [
        f"- [{r.item_id}] {r.member_id}: {r.role} ({r.confidence or 'weak'})"
        for r in profile.roles[:MAX_ROLES]
    ]
    return "\n".join(
        [
            "TOPICS",
            *(topics or ["(none)"]),
            "DECISIONS",
            *(decisions or ["(none)"]),
            "OPEN DISSENT",
            *(dissent or ["(none)"]),
            "ROLES",
            *(roles or ["(none)"]),
        ]
    )


def _shown_summary(summary: str) -> str:
    """``summary`` cut to :data:`TOPIC_SUMMARY_SHOWN` characters, the cut marked."""
    if len(summary) <= TOPIC_SUMMARY_SHOWN:
        return summary
    return summary[:TOPIC_SUMMARY_SHOWN].rstrip() + "…"


def _decision_line(item: GroupProfileItem) -> str:
    parts = [f"- [{item.item_id}]"]
    if item.topic_id:
        parts.append(f"(topic {item.topic_id})")
    parts.append(item.statement)
    who = [
        f"{label} {', '.join(ids)}"
        for label, ids in (
            ("decided by", item.decided_by),
            ("agreed by", item.agreed_by),
            ("assigned to", item.assignees),
            ("applies to", item.applies_to),
        )
        if ids
    ]
    if who:
        parts.append(f"-- {'; '.join(who)}")
    return " ".join(parts)


def render_segments(segments: Sequence[Segment], roster: Roster) -> str:
    """Each segment under its marker, one ``[id] time speaker: text`` line per message.

    Speakers carry their roster names (an assistant as ``"<name>
    (assistant)"``) and their member ids, ``Chen (u_chen)``; a message's line
    breaks become spaces so every message is one line.
    """
    blocks: list[str] = []
    for segment in segments:
        lines = [f"=== SEGMENT {segment.memcell_id} ==="]
        for item in label_cell(segment.cell, roster).items:
            if not isinstance(item, ChatMessage):
                continue
            when = from_timestamp_ms(item.timestamp).strftime("%Y-%m-%d %H:%M")
            text = " ".join(str(item.content or "").split())
            lines.append(f"[{item.id}] {when} {speaker_with_id(item)}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
