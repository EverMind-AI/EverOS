"""RosterManager -- a group's meta and roster, authorised and folded.

The domain half of the group routes (docs/group-memory.md, section 5.1) and of
the group fields of ``/memory/add`` (section 5.2). It decides who may change
what (:mod:`everos.memory.access`), which events a request appends -- the
asker as owner of a group it creates, an implicit join for a sender who is not
a member -- and lets the store refuse what does not fold (409). One group's
changes are serialised on the ``group`` lock, so the roster an authorisation
read is the roster the write extends.

Lock order, everywhere: ``session`` (sorted, when several) -> ``group`` ->
``owner`` (the store's markdown lock).
"""

from __future__ import annotations

__all__ = [
    "BatchAdmission",
    "BatchMessage",
    "GroupMetaUpdate",
    "RosterChange",
    "RosterManager",
    "implicit_joins",
]

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from everos.core.errors import (
    ForbiddenError,
    InvalidInputError,
    RosterConflictError,
)
from everos.core.locks import LOCKS, owner_key
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory.access import authorize_manage, authorize_roster_change
from everos.memory.models import (
    Group,
    HistoryVisibleFrom,
    Roster,
    RosterEvent,
    RosterSource,
)
from everos.memory.ports import GroupRecord, GroupStore, Scope, SessionKey

logger = get_logger(__name__)


@dataclass(frozen=True)
class GroupMetaUpdate:
    """Meta fields a ``PUT /groups/{id}`` sets; ``None`` keeps the stored value."""

    name: str | None = None
    description: str | None = None
    created_at: int | None = None
    extra: dict[str, Any] | None = None
    history_visible_from: HistoryVisibleFrom | None = None


@dataclass(frozen=True)
class RosterChange:
    """What a roster write did."""

    record: GroupRecord
    applied: int
    created: bool = False


@dataclass(frozen=True)
class BatchMessage:
    """The part of one ``/memory/add`` message the roster cares about."""

    sender_id: str
    role: str
    at: int
    sender_name: str | None = None


@dataclass(frozen=True)
class BatchAdmission:
    """The group side of one admitted batch: the record and what was appended."""

    record: GroupRecord
    created: bool
    appended: list[RosterEvent] = field(default_factory=list)


def implicit_joins(
    roster: Roster, messages: Sequence[BatchMessage]
) -> list[RosterEvent]:
    """Joins for senders the roster does not know at their message's time.

    A sender with no interval opens one at their first message here
    (``source = implicit``), as does a former member whose message comes after
    their last leave (a rejoin). A message from a current member that predates
    their join, or from a former member inside or before their old stay, is
    accepted without a roster change: the roster is the authority on windows,
    and a message is not evidence enough to rewrite it. Tool rows are not
    members.
    """
    joined: set[str] = set()
    events: list[RosterEvent] = []
    for message in sorted(messages, key=lambda m: m.at):
        if message.role == "tool" or message.sender_id in joined:
            continue
        sender = message.sender_id
        if roster.interval_at(sender, message.at) is not None:
            continue
        if roster.current(sender) is not None:
            continue
        ends = [i.left_at for i in roster.intervals_of(sender) if i.left_at is not None]
        if ends and message.at < max(ends):
            continue
        assistant = message.role == "assistant"
        events.append(
            RosterEvent(
                op="join",
                member_id=sender,
                at=message.at,
                source="implicit",
                kind="assistant" if assistant else "human",
                role="assistant" if assistant else "member",
                display_name=message.sender_name or sender,
            )
        )
        joined.add(sender)
    return events


def _now_ms() -> int:
    return to_timestamp_ms(get_utc_now())


def _with_owner(
    events: Sequence[RosterEvent],
    asker_id: str | None,
    *,
    at: int,
    source: RosterSource,
) -> list[RosterEvent]:
    """``events``, plus the asker joining as owner when none of them names one."""
    out = list(events)
    if asker_id is None or any(e.op == "join" and e.role == "owner" for e in out):
        return out
    mine = [e for e in out if e.member_id == asker_id and e.op == "join"]
    if mine:
        # The asker is in the roster already: promote that join, keep its time.
        first = mine[0]
        out[out.index(first)] = first.model_copy(update={"role": "owner"})
        return out
    return [
        RosterEvent(op="join", member_id=asker_id, role="owner", at=at, source=source),
        *out,
    ]


class RosterManager:
    """Group meta and roster changes over a :class:`GroupStore`.

    Args:
        store: The groups of the memory root.
        implicit_create: ``[groups] implicit_create`` -- an unknown
            ``group_id`` on ``/memory/add`` creates the group.
        implicit_join: ``[groups] implicit_join`` -- an unknown sender joins
            at their first message; ``False`` refuses the batch (422).
        clock: Epoch-ms "now", for events and groups that name no time.
    """

    def __init__(
        self,
        store: GroupStore,
        *,
        implicit_create: bool = True,
        implicit_join: bool = True,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        self._store = store
        self._implicit_create = implicit_create
        self._implicit_join = implicit_join
        self._clock = clock

    @property
    def store(self) -> GroupStore:
        return self._store

    async def read(self, scope: Scope, *, asker_id: str | None) -> GroupRecord:
        """The group, for a current member (``GET /groups/{id}``).

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: Not a current member -- or no such group, which
                answers the same.
        """
        if asker_id is None:
            raise InvalidInputError("asker_id is required to read a group")
        record = await self._store.get(scope)
        if record is None or not record.roster.is_current_member(asker_id):
            raise ForbiddenError("the asker is not a current member of the group")
        return record

    async def upsert(
        self,
        scope: Scope,
        *,
        asker_id: str | None,
        meta: GroupMetaUpdate,
        roster: Sequence[RosterEvent] = (),
    ) -> RosterChange:
        """Create the group, or update its meta and append ``roster``.

        A new group's roster that names no owner makes the asker its owner.
        An existing group's meta and roster are changed by an owner or admin.

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: The asker may not manage the existing group.
            RosterConflictError: The roster does not fold.
        """
        if asker_id is None:
            raise InvalidInputError("asker_id is required to create or update a group")
        async with LOCKS.hold("group", _key(scope)):
            existing = await self._store.get(scope)
            if existing is None:
                created_at = (
                    meta.created_at if meta.created_at is not None else self._clock()
                )
                group = Group(
                    group_id=scope.owner_id,
                    name=meta.name or "",
                    description=meta.description or "",
                    created_at=created_at,
                    created_by=asker_id,
                    extra=dict(meta.extra or {}),
                    history_visible_from=meta.history_visible_from or "join",
                )
                events = _with_owner(roster, asker_id, at=created_at, source="api")
                record = await self._store.save(scope, group=group, events=events)
                return RosterChange(record=record, applied=len(events), created=True)
            authorize_manage(existing.roster, asker_id)
            if roster:
                authorize_roster_change(existing.roster, asker_id, roster)
            changes = {
                name: value
                for name, value in (
                    ("name", meta.name),
                    ("description", meta.description),
                    ("extra", meta.extra),
                    ("history_visible_from", meta.history_visible_from),
                )
                if value is not None
            }
            group = existing.group.model_copy(update=changes)
            record = await self._store.save(scope, group=group, events=list(roster))
            return RosterChange(record=record, applied=len(roster))

    async def change(
        self, scope: Scope, *, asker_id: str | None, events: Sequence[RosterEvent]
    ) -> RosterChange:
        """Append roster events (``POST /groups/{id}/roster``).

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: Not a current member (or no such group), or not
                allowed to send one of the events.
            RosterConflictError: The events do not fold.
        """
        if asker_id is None:
            raise InvalidInputError("asker_id is required to change a roster")
        async with LOCKS.hold("group", _key(scope)):
            existing = await self._store.get(scope)
            if existing is None:
                raise ForbiddenError("the asker is not a current member of the group")
            authorize_roster_change(existing.roster, asker_id, events)
            record = await self._store.save(
                scope, group=existing.group, events=list(events)
            )
        return RosterChange(record=record, applied=len(events))

    async def check_binding(self, session: SessionKey, group_id: str) -> None:
        """Raise when ``session`` is bound to a different group (409, D18)."""
        bound = await self._store.binding(session)
        if bound is not None and bound != group_id:
            raise RosterConflictError(
                f"session {session.session_id!r} belongs to another group"
            )

    async def admit_batch(
        self,
        scope: Scope,
        *,
        session: SessionKey,
        asker_id: str | None,
        roster: Sequence[RosterEvent],
        messages: Sequence[BatchMessage],
    ) -> BatchAdmission:
        """The group side of one ``/memory/add`` batch (section 5.2).

        Under the group lock: refuse a session bound elsewhere; create the
        group if it is unknown (its first message's time is its creation
        time); apply the batch's inline roster events (an existing group's
        need an owner or admin as asker); open implicit intervals for senders
        who are not members; bind the session.

        Raises:
            RosterConflictError: The session belongs to another group, or the
                roster does not fold.
            ForbiddenError: Inline roster events for an existing group from an
                asker who may not send them.
            InvalidInputError: An unknown group with ``implicit_create`` off,
                or an unknown sender with ``implicit_join`` off.
        """
        first = min((m.at for m in messages), default=None)
        async with LOCKS.hold("group", _key(scope)):
            await self.check_binding(session, scope.owner_id)
            existing = await self._store.get(scope)
            appended: list[RosterEvent] = []
            if existing is None:
                if not self._implicit_create:
                    raise InvalidInputError(
                        f"unknown group {scope.owner_id!r} ([groups] implicit_create "
                        "is off)"
                    )
                created_at = first if first is not None else self._clock()
                group = Group(
                    group_id=scope.owner_id, created_at=created_at, created_by=asker_id
                )
                appended = _with_owner(roster, asker_id, at=created_at, source="inline")
                record = await self._store.save(scope, group=group, events=appended)
            else:
                record = existing
                if roster:
                    authorize_roster_change(existing.roster, asker_id, roster)
                    appended = list(roster)
                    record = await self._store.save(
                        scope, group=existing.group, events=appended
                    )
            joins = implicit_joins(record.roster, messages)
            if joins:
                if not self._implicit_join:
                    raise InvalidInputError(
                        "senders who are not members: "
                        + ", ".join(sorted({e.member_id for e in joins}))
                        + " ([groups] implicit_join is off)"
                    )
                record = await self._store.save(scope, group=record.group, events=joins)
                appended += joins
            await self._store.bind(
                session,
                scope.owner_id,
                at=first if first is not None else self._clock(),
            )
        if appended:
            logger.info(
                "groups.batch_roster_applied",
                app_id=scope.app_id,
                project_id=scope.project_id,
                group_id=scope.owner_id,
                events=len(appended),
                created=existing is None,
            )
        return BatchAdmission(
            record=record, created=existing is None, appended=appended
        )


def _key(scope: Scope) -> str:
    return owner_key(scope.app_id, scope.project_id, "group", scope.owner_id)
