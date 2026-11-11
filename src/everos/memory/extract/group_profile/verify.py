"""Code checks on the update call's operations, and the items they become.

docs/group-memory.md, section 7.5. Every operation is checked; one that fails
any check is dropped -- never repaired -- and counted by reason
(``everos_group_profile_ops_dropped_total``):

* ``no_evidence``: it cites nothing;
* ``unknown_segment`` / ``unknown_message``: a cited segment is not one of
  the batch's, or the message is not in it;
* ``quote_not_verbatim``: a quote is not part of that message's text after
  whitespace normalisation, curly quotation marks read as straight ones
  (:func:`everos.memory.groups.verbatim_part`; an empty quote is not one
  either);
* ``assistant_only_decision``: a ``decision.*`` operation whose evidence
  holds no human member's message;
* ``ungrounded_member``: ``decided_by`` / ``agreed_by`` / ``held_by`` /
  ``member_id`` names someone none of the cited messages is from -- the
  speaker is the message's sender, never the model's say-so;
* ``unknown_member``: ``assignees`` / ``applies_to`` names someone not in the
  roster, or a role or dissent is given to an assistant;
* ``unknown_target`` / ``unknown_reference``: the id it updates or
  supersedes, or the topic / item it refers to, is not one of the rendered
  current items;
* ``invalid_field``: a value outside its set (a topic status, a role, a
  confidence) or a required text left empty;
* ``duplicate``: a second operation on the same target in one reply, or a
  role the member already holds.

Speakers and times are copied from the cited messages, and a quote is stored
as the message's own words; the attribution fields and the span of an item
come from its verified evidence alone.
"""

from __future__ import annotations

__all__ = ["DROP_REASONS", "Verified", "drop_summary", "mint_topic_id", "verify"]

import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast, get_args

from everalgo.types import ChatMessage

from everos.core.observability.metrics import Counter as MetricCounter
from everos.memory.groups import verbatim_part
from everos.memory.models import (
    GroupProfileItem,
    MemberKind,
    ObservedRole,
    Origin,
    ProfileSource,
    Roster,
    TopicStatus,
)

from .fold import GroupProfileView
from .prompt import Segment
from .reply import Operation

DROP_REASONS: tuple[str, ...] = (
    "no_evidence",
    "unknown_segment",
    "unknown_message",
    "quote_not_verbatim",
    "assistant_only_decision",
    "ungrounded_member",
    "unknown_member",
    "unknown_target",
    "unknown_reference",
    "invalid_field",
    "duplicate",
)

_ops_total = MetricCounter(
    name="group_profile_ops_total",
    description="Group-profile update operations the model proposed, by outcome.",
    labelnames=("outcome",),
    namespace="everos",
)
_ops_dropped = MetricCounter(
    name="group_profile_ops_dropped_total",
    description="Group-profile update operations dropped by the verifier, by reason.",
    labelnames=("reason",),
    namespace="everos",
)

_TOPIC_STATUSES = frozenset(get_args(TopicStatus))
_ROLES = frozenset(get_args(ObservedRole))
_CONFIDENCES = frozenset({"strong", "weak"})


class _DroppedError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Verified:
    """What survived the checks, and what did not.

    Attributes:
        items: The new items, in reply order, ready to append.
        dropped: Dropped operations by reason.
    """

    items: list[GroupProfileItem] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)

    @property
    def proposed(self) -> int:
        return len(self.items) + sum(self.dropped.values())


def mint_topic_id() -> str:
    """A new topic's id; the model never mints one (a rename keeps it)."""
    return f"tp_{uuid.uuid4().hex[:8]}"


def verify(
    operations: Sequence[Operation],
    *,
    group_id: str,
    segments: Sequence[Segment],
    roster: Roster,
    profile: GroupProfileView,
    observed_at: int,
    shown_topics: Sequence[GroupProfileItem] | None = None,
    shown_decisions: Sequence[GroupProfileItem] | None = None,
) -> Verified:
    """Check ``operations`` and build the items of the ones that pass.

    Args:
        operations: The reply's operations.
        group_id: The group (the items' owner).
        segments: The batch the call was shown.
        roster: The group's roster.
        profile: The current profile the call was built from.
        observed_at: The batch's last segment time, stamped on every item.
        shown_topics / shown_decisions: The topics and decisions the prompt
            rendered (the caps cut them); ``None`` means all of ``profile``'s.

    Returns:
        The items to append and the drop counts; the counts also go to the
        metrics.
    """
    checker = _Checker(
        group_id=group_id,
        segments=segments,
        roster=roster,
        profile=profile,
        observed_at=observed_at,
        topics=list(profile.topics if shown_topics is None else shown_topics),
        decisions=list(
            profile.decisions if shown_decisions is None else shown_decisions
        ),
    )
    result = Verified()
    for operation in operations:
        try:
            result.items.append(checker.check(operation))
        except _DroppedError as drop:
            result.dropped[drop.reason] += 1
    _ops_total.labels(outcome="kept").inc(len(result.items))
    _ops_total.labels(outcome="dropped").inc(sum(result.dropped.values()))
    for reason, count in result.dropped.items():
        _ops_dropped.labels(reason=reason).inc(count)
    return result


class _Checker:
    """The checks of one reply, holding the batch and the rendered ids."""

    def __init__(
        self,
        *,
        group_id: str,
        segments: Sequence[Segment],
        roster: Roster,
        profile: GroupProfileView,
        observed_at: int,
        topics: Sequence[GroupProfileItem],
        decisions: Sequence[GroupProfileItem],
    ) -> None:
        self._group_id = group_id
        self._roster = roster
        self._observed_at = observed_at
        self._messages: dict[str, dict[str, ChatMessage]] = {
            s.memcell_id: {
                item.id: item for item in s.cell.items if isinstance(item, ChatMessage)
            }
            for s in segments
        }
        self._topics = {t.topic_id: t for t in topics if t.topic_id}
        self._decisions = {d.item_id: d for d in decisions if d.item_id}
        self._dissent = {d.item_id: d for d in profile.dissent if d.item_id}
        self._roles = {r.item_id: r for r in profile.roles if r.item_id}
        self._held = {(r.member_id, r.role) for r in profile.roles}
        self._targets: set[str] = set()

    # ── one operation ─────────────────────────────────────────────────

    def check(self, op: Operation) -> GroupProfileItem:
        cited = self._sources(op)
        sources = [source for source, _ in cited]
        kinds = {source.speaker_id: kind for source, kind in cited}
        speakers = list(kinds)
        humans = [s for s in speakers if kinds[s] == "human"]
        base = {
            "owner_id": self._group_id,
            "sources": sources,
            "sender_ids": speakers,
            "origin": _origin(set(kinds.values())),
            "source_start": min(s.at for s in sources),
            "source_end": max(s.at for s in sources),
            "observed_at": self._observed_at,
        }
        if op.op.startswith("topic."):
            return self._topic(op, base, humans)
        if op.op.startswith("decision."):
            return self._decision(op, base, speakers, humans)
        if op.op.startswith("dissent."):
            return self._dissent_item(op, base, speakers, humans)
        return self._role(op, base, speakers, humans)

    def _topic(
        self, op: Operation, base: dict[str, object], humans: list[str]
    ) -> GroupProfileItem:
        if op.status not in _TOPIC_STATUSES:
            raise _DroppedError("invalid_field")
        if op.op == "topic.new":
            if not op.name.strip():
                raise _DroppedError("invalid_field")
            topic_id, old = mint_topic_id(), None
        else:
            old = self._topics.get(op.target)
            if old is None:
                raise _DroppedError("unknown_target")
            self._claim(op.target)
            topic_id = old.topic_id
        return GroupProfileItem(
            **base,  # type: ignore[arg-type]
            kind="topic",
            topic_id=topic_id,
            name=op.name.strip() or (old.name if old else ""),
            summary=op.summary.strip() or (old.summary if old else ""),
            topic_status=cast(TopicStatus, op.status),
            supersedes=old.item_id if old else None,
            subject_ids=humans,
        )

    def _decision(
        self,
        op: Operation,
        base: dict[str, object],
        speakers: list[str],
        humans: list[str],
    ) -> GroupProfileItem:
        if not humans:
            raise _DroppedError("assistant_only_decision")
        if not op.statement.strip():
            raise _DroppedError("invalid_field")
        self._grounded([*op.decided_by, *op.agreed_by], speakers)
        self._members([*op.assignees, *op.applies_to])
        old = None
        if op.op == "decision.supersede":
            old = self._decisions.get(op.target)
            if old is None:
                raise _DroppedError("unknown_target")
            self._claim(op.target)
        topic_id = op.topic_id or (old.topic_id if old else None)
        if topic_id and topic_id not in self._topics:
            raise _DroppedError("unknown_reference")
        people = [*op.decided_by, *op.agreed_by, *op.assignees, *op.applies_to]
        return GroupProfileItem(
            **base,  # type: ignore[arg-type]
            kind="decision",
            statement=op.statement.strip(),
            topic_id=topic_id or None,
            decided_by=list(dict.fromkeys(op.decided_by)),
            agreed_by=list(dict.fromkeys(op.agreed_by)),
            assignees=list(dict.fromkeys(op.assignees)),
            applies_to=list(dict.fromkeys(op.applies_to)),
            supersedes=old.item_id if old else None,
            subject_ids=list(dict.fromkeys(people)) or humans,
        )

    def _dissent_item(
        self,
        op: Operation,
        base: dict[str, object],
        speakers: list[str],
        humans: list[str],
    ) -> GroupProfileItem:
        if op.op == "dissent.open":
            if not op.statement.strip() or not op.held_by:
                raise _DroppedError("invalid_field")
            self._grounded(op.held_by, speakers)
            if any(self._kind_now(m) != "human" for m in op.held_by):
                raise _DroppedError("unknown_member")
            if (
                op.about
                and op.about not in self._decisions
                and (op.about not in self._topics)
            ):
                raise _DroppedError("unknown_reference")
            return GroupProfileItem(
                **base,  # type: ignore[arg-type]
                kind="dissent",
                statement=op.statement.strip(),
                held_by=list(dict.fromkeys(op.held_by)),
                about=op.about or None,
                subject_ids=list(dict.fromkeys(op.held_by)),
            )
        old = self._dissent.get(op.target)
        if old is None:
            raise _DroppedError("unknown_target")
        if not humans:
            raise _DroppedError("ungrounded_member")
        self._claim(op.target)
        return GroupProfileItem(
            **base,  # type: ignore[arg-type]
            kind="dissent",
            statement=op.statement.strip() or old.statement,
            held_by=list(old.held_by),
            about=old.about,
            resolved=True,
            resolved_by=humans,
            supersedes=old.item_id,
            subject_ids=list(old.held_by) or humans,
        )

    def _role(
        self,
        op: Operation,
        base: dict[str, object],
        speakers: list[str],
        humans: list[str],
    ) -> GroupProfileItem:
        if op.op == "role.retire":
            old = self._roles.get(op.target)
            if old is None:
                raise _DroppedError("unknown_target")
            self._claim(op.target)
            return GroupProfileItem(
                **base,  # type: ignore[arg-type]
                kind="role",
                member_id=old.member_id,
                role=old.role,
                confidence=old.confidence,
                retired=True,
                supersedes=old.item_id,
                subject_ids=[old.member_id] if old.member_id else humans,
            )
        if op.role not in _ROLES or op.confidence not in _CONFIDENCES:
            raise _DroppedError("invalid_field")
        self._grounded([op.member_id], speakers)
        if self._kind_now(op.member_id) != "human":
            raise _DroppedError("unknown_member")
        if (op.member_id, op.role) in self._held:
            raise _DroppedError("duplicate")
        self._held.add((op.member_id, op.role))
        return GroupProfileItem(
            **base,  # type: ignore[arg-type]
            kind="role",
            member_id=op.member_id,
            role=cast(ObservedRole, op.role),
            confidence=op.confidence,  # type: ignore[arg-type]
            subject_ids=[op.member_id],
        )

    # ── the checks ────────────────────────────────────────────────────

    def _sources(self, op: Operation) -> list[tuple[ProfileSource, MemberKind]]:
        """The verified sources, each with its speaker's kind at that message."""
        if not op.evidence:
            raise _DroppedError("no_evidence")
        sources: list[tuple[ProfileSource, MemberKind]] = []
        for evidence in op.evidence:
            messages = self._messages.get(evidence.segment)
            if messages is None:
                raise _DroppedError("unknown_segment")
            message = messages.get(evidence.message)
            if message is None:
                raise _DroppedError("unknown_message")
            quote = verbatim_part(evidence.quote, str(message.content or ""))
            if quote is None:
                raise _DroppedError("quote_not_verbatim")
            source = ProfileSource(
                memcell_id=evidence.segment,
                message_id=message.id,
                speaker_id=message.sender_id,
                at=message.timestamp,
                quote=quote,
            )
            sources.append((source, self._kind(message)))
        return sources

    def _grounded(self, members: Sequence[str], speakers: Sequence[str]) -> None:
        if any(m not in speakers for m in members):
            raise _DroppedError("ungrounded_member")

    def _members(self, members: Sequence[str]) -> None:
        if any(self._kind_now(m) is None for m in members):
            raise _DroppedError("unknown_member")

    def _claim(self, target: str) -> None:
        if target in self._targets:
            raise _DroppedError("duplicate")
        self._targets.add(target)

    def _kind(self, message: ChatMessage) -> MemberKind:
        """The sender's roster kind at the message; its role without a roster entry."""
        interval = self._roster.interval_at(message.sender_id, message.timestamp)
        if interval is not None:
            return interval.kind
        known = self._kind_now(message.sender_id)
        if known is not None:
            return known
        return "assistant" if message.role == "assistant" else "human"

    def _kind_now(self, member_id: str) -> MemberKind | None:
        """The member's kind by their latest interval; ``None`` if never a member."""
        intervals = self._roster.intervals_of(member_id)
        if not intervals:
            return None
        return max(intervals, key=lambda i: i.joined_at).kind


def _origin(kinds: set[MemberKind]) -> Origin | None:
    if not kinds:
        return None
    if kinds == {"human"}:
        return "member"
    if kinds == {"assistant"}:
        return "assistant"
    return "mixed"


def drop_summary(dropped: Mapping[str, int]) -> dict[str, int]:
    """The drop counts in :data:`DROP_REASONS` order, zeros left out."""
    return {r: dropped[r] for r in DROP_REASONS if dropped.get(r)}
