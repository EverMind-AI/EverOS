"""The group use cases behind ``/groups``: create or update, change the roster, read.

Each step resolves the asker (the gateway's identity header wins and must
agree with the body, docs/group-memory.md section 9.2), hands the change to
the domain's :class:`~everos.memory.groups.RosterManager`, and shapes the
result into plain service values -- the entry points never see a domain type.

A roster change first cuts the open buffers of the group's sessions (D18): the
messages buffered before a membership change become segments before the change
is applied, so no segment spans it. Correctness does not depend on the cut
(every read checks ``source_start`` / ``source_end`` against the windows); the
cut is what keeps a new member's first segment visible to them.
"""

from __future__ import annotations

__all__ = [
    "GroupUpsertResult",
    "GroupUseCases",
    "GroupView",
    "RosterChangeResult",
    "RosterEventInput",
    "to_events",
]

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from everos.memory.access import (
    authorize_manage,
    authorize_roster_change,
    resolve_identity,
)
from everos.memory.groups import GroupMetaUpdate, RosterChange, RosterManager
from everos.memory.models import RosterEvent, RosterSource
from everos.memory.ports import GroupRecord, Scope

if TYPE_CHECKING:
    from ._memorize import Memorizer


@dataclass(frozen=True)
class RosterEventInput:
    """One roster event as a caller sends it; ``at`` defaults to now."""

    op: str
    member_id: str
    kind: str | None = None
    display_name: str | None = None
    aliases: list[str] | None = None
    role: str | None = None
    title: str | None = None
    at: int | None = None


@dataclass(frozen=True)
class GroupView:
    """A group as a route returns it; ``roster`` holds interval mappings."""

    group_id: str
    app_id: str
    project_id: str
    name: str
    description: str
    created_at: int
    created_by: str | None
    history_visible_from: str
    extra: dict[str, Any] = field(default_factory=dict)
    roster: list[dict[str, Any]] = field(default_factory=list)
    deliverer_id: str | None = None


@dataclass(frozen=True)
class GroupUpsertResult:
    created: bool
    events_applied: int
    group: GroupView


@dataclass(frozen=True)
class RosterChangeResult:
    events_applied: int
    group: GroupView


def to_events(
    inputs: Sequence[RosterEventInput | dict[str, Any]],
    *,
    source: RosterSource,
    now: int,
) -> list[RosterEvent]:
    """Domain events for ``inputs``; an event without ``at`` happens ``now``."""
    events: list[RosterEvent] = []
    for raw in inputs:
        data = dict(raw) if isinstance(raw, dict) else dict(raw.__dict__)
        if data.get("at") is None:
            data["at"] = now
        events.append(RosterEvent(source=source, **data))
    return events


def group_view(
    record: GroupRecord, scope: Scope, *, include_history: bool = False
) -> GroupView:
    """The route-facing view of ``record``: current members, or every interval."""
    intervals = (
        record.roster.intervals if include_history else record.roster.current_members()
    )
    group = record.group
    return GroupView(
        group_id=group.group_id,
        app_id=scope.app_id,
        project_id=scope.project_id,
        name=group.name,
        description=group.description,
        created_at=group.created_at,
        created_by=group.created_by,
        history_visible_from=group.history_visible_from,
        extra=dict(group.extra),
        roster=[interval.model_dump() for interval in intervals],
        deliverer_id=group.deliverer_id,
    )


def _scope(app_id: str, project_id: str, group_id: str) -> Scope:
    return Scope(
        app_id=app_id, project_id=project_id, owner_id=group_id, owner_type="group"
    )


class GroupUseCases:
    """The three group use cases over the roster manager and the memorizer.

    Args:
        rosters: The domain's roster rules over the group store.
        memorizer: Cuts the group's open session buffers before a change.
        clock: Epoch-ms "now" for events that name no time.
    """

    def __init__(
        self,
        *,
        rosters: RosterManager,
        memorizer: Memorizer,
        clock: Callable[[], int],
    ) -> None:
        self._rosters = rosters
        self._memorizer = memorizer
        self._clock = clock

    async def upsert(
        self,
        *,
        app_id: str,
        project_id: str,
        group_id: str,
        asker_id: str | None,
        asserted_asker: str | None,
        name: str | None,
        description: str | None,
        created_at: int | None,
        history_visible_from: str | None,
        extra: dict[str, Any] | None,
        roster: Sequence[RosterEventInput],
        deliverer_id: str | None = None,
        clear_deliverer: bool = False,
    ) -> GroupUpsertResult:
        """Create the group or update its meta; ``deliverer_id`` sets the
        group's deliverer and ``clear_deliverer`` removes it (section 5.1)."""
        identity = resolve_identity(header=asserted_asker, body_asker=asker_id)
        scope = _scope(app_id, project_id, group_id)
        events = to_events(roster, source="api", now=self._clock())
        meta = GroupMetaUpdate(
            name=name,
            description=description,
            created_at=created_at,
            extra=extra,
            history_visible_from=history_visible_from,  # type: ignore[arg-type]
            deliverer_id=deliverer_id,
            clear_deliverer=clear_deliverer,
        )

        async def apply() -> RosterChange:
            return await self._rosters.upsert(
                scope, asker_id=identity.asker_id, meta=meta, roster=events
            )

        def precheck(record: GroupRecord) -> None:
            authorize_manage(record.roster, identity.asker_id)
            authorize_roster_change(record.roster, identity.asker_id, events)

        change = await self._with_buffers_closed(scope, events, apply, precheck)
        return GroupUpsertResult(
            created=change.created,
            events_applied=change.applied,
            group=group_view(change.record, scope),
        )

    async def change(
        self,
        *,
        app_id: str,
        project_id: str,
        group_id: str,
        asker_id: str | None,
        asserted_asker: str | None,
        events: Sequence[RosterEventInput],
    ) -> RosterChangeResult:
        identity = resolve_identity(header=asserted_asker, body_asker=asker_id)
        scope = _scope(app_id, project_id, group_id)
        domain_events = to_events(events, source="api", now=self._clock())

        async def apply() -> RosterChange:
            return await self._rosters.change(
                scope, asker_id=identity.asker_id, events=domain_events
            )

        def precheck(record: GroupRecord) -> None:
            authorize_roster_change(record.roster, identity.asker_id, domain_events)

        change = await self._with_buffers_closed(scope, domain_events, apply, precheck)
        return RosterChangeResult(
            events_applied=change.applied, group=group_view(change.record, scope)
        )

    async def read(
        self,
        *,
        app_id: str,
        project_id: str,
        group_id: str,
        asker_id: str | None,
        asserted_asker: str | None,
        include_history: bool,
    ) -> GroupView:
        identity = resolve_identity(header=asserted_asker, body_asker=asker_id)
        scope = _scope(app_id, project_id, group_id)
        record = await self._rosters.read(scope, asker_id=identity.asker_id)
        return group_view(record, scope, include_history=include_history)

    async def _with_buffers_closed(
        self,
        scope: Scope,
        events: Sequence[RosterEvent],
        apply: Callable[[], Awaitable[RosterChange]],
        precheck: Callable[[GroupRecord], None],
    ) -> RosterChange:
        """Run ``apply`` with the group's session buffers cut first (D18).

        ``precheck`` refuses an unauthorised change before any buffer is cut,
        as does an implicit group, which takes no change at all; ``apply``
        checks again under the group lock, which is authoritative.
        """
        self._rosters.check_declared(scope)
        if not events:
            return await apply()
        record = await self._rosters.store.get(scope)
        if record is None or not record.sessions:
            return await apply()
        precheck(record)
        sessions = record.sessions
        return await self._memorizer.with_buffers_closed(
            app_id=scope.app_id,
            project_id=scope.project_id,
            sessions=sessions,
            change=apply,
        )
