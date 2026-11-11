"""The group-profile update prompt, ``chat.group_profile_update`` (section 7.5).

One call per batch of segments. It is shown:

* who is who: every member of the roster by id, with display name, access
  role and title, the assistants marked, and the roster block's attribution
  rule (an assistant's words are its suggestion, never a group decision);
* the current profile, every item with its id: the active topics (at most
  :data:`MAX_TOPICS`, most recently active first -- replaced topics stay
  stored and are not shown), the current decisions, open dissent and active
  roles;
* the new segments, each opened by ``=== SEGMENT <memcell_id> ===``, one line
  per message: ``[<message_id>] <time> <speaker>: <text>``.

It answers with operations (:mod:`.reply`), each citing evidence as
``(segment, message, quote)``. Code checks every one (:mod:`.verify`).

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
from everos.memory.extract.pipeline.group_render import label_cell
from everos.memory.models import GroupProfileItem, MembershipInterval, Roster
from everos.memory.prompts import PromptTemplate

from .fold import GroupProfileView

MAX_TOPICS = 20
"""Active topics shown to the model, most recently active first."""
MAX_DECISIONS = 40
MAX_DISSENT = 20
MAX_ROLES = 30

GROUP_PROFILE_UPDATE = PromptTemplate(
    id="chat.group_profile_update",
    version=1,
    parts=(
        # 0: the system prompt; {members} and {language_rule} are filled in.
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
- topic.new: a substantive thread the profile does not have yet. Fill name, \
summary and status.
- topic.update: a listed topic whose summary or status changed. target = its \
topic id (tp_...). Fill name, summary and status as they are now.
- decision.new: something the members decided or agreed to do. Fill \
statement, decided_by (who made the decision), agreed_by (who agreed), \
assignees (who will carry it out), applies_to (the members it applies to; \
empty when it applies to the whole group) and topic_id (a listed topic id, or \
empty).
- decision.supersede: a new decision that replaces a listed decision. target \
= the id of the decision it replaces. Fill the same fields as decision.new.
- dissent.open: a member disagrees with a decision or a topic's direction and \
has not been brought round. Fill statement (what they object to and why), \
held_by and about (a listed decision id or topic id, or empty).
- dissent.resolve: a listed open dissent is settled. target = its id; \
statement = how it was settled.
- role.add: a member clearly plays one of the roles below. Fill member_id, \
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
2. People. Use member ids from <members> in decided_by, agreed_by, \
assignees, applies_to, held_by and member_id. decided_by, agreed_by, held_by \
and member_id must be people whose own messages the operation cites.
3. The assistant. A decision needs at least one member's own message that \
makes or confirms it; the assistant's suggestion alone is not a decision. \
Never give the assistant a role or a dissent.
4. Topics. A topic is a substantive thread: a plan, a problem, a piece of \
work, a question the group works through. Scheduling chatter, greetings, \
thanks, bot notices and one-word confirmations are not topics. Status: \
exploring (being discussed), disagreement (members disagree), consensus \
(members agree), implemented (done).
5. Ids. Use only ids that appear in <profile>. Never invent an id. Leave \
target, topic_id and about empty when they do not apply, and every field an \
operation does not use empty ("" or []).
6. Only what changed. Do not repeat what the profile already says. If \
nothing changed, reply {{"operations": []}}.
7. {language_rule}
</rules>

Reply with JSON only, in this shape:
{{"operations": [{{"op": "decision.new", "target": "", "topic_id": "", \
"name": "", "summary": "", "status": "", "statement": "", "decided_by": [], \
"agreed_by": [], "assignees": [], "applies_to": [], "held_by": [], \
"about": "", "member_id": "", "role": "", "confidence": "", "evidence": \
[{{"segment": "", "message": "", "quote": ""}}]}}]}}""",
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
                members=render_members(roster, segments), language_rule=rule
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
    """The current items with their ids, by kind; ``(empty)`` sections kept."""
    topics = [
        f"- [{t.topic_id}] {t.name} ({t.topic_status or 'exploring'}): {t.summary}"
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
    (assistant)"``); a message's line breaks become spaces so every message is
    one line.
    """
    blocks: list[str] = []
    for segment in segments:
        lines = [f"=== SEGMENT {segment.memcell_id} ==="]
        for item in label_cell(segment.cell, roster).items:
            if not isinstance(item, ChatMessage):
                continue
            when = from_timestamp_ms(item.timestamp).strftime("%Y-%m-%d %H:%M")
            speaker = item.sender_name or item.sender_id
            text = " ".join(str(item.content or "").split())
            lines.append(f"[{item.id}] {when} {speaker}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
