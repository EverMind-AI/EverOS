"""catch_up_profiles strategy -- every profile on a cadence, up to date at once.

The profiles follow a cadence, not every segment: a user's profile reads its
queue every ``[profile] update_every`` segments or ``[profile] idle_seconds``
after the user's last one, a group's member profiles and group profile read
their batches after ``[groups] group_profile_batch`` new segments or
``group_profile_idle_seconds`` without one, and a flush reads none of them
unless ``[profile] update_on_flush`` / ``[groups] profile_on_flush`` is set.
Whoever needs them current at once -- an evaluation about to search what it
just ingested, an operator -- triggers this strategy::

    POST /api/v2/ome/trigger  {"name": "catch_up_profiles", "timeout": 1800}

It runs on that :class:`~everos.memory.ports.ManualTick` only and asks every
:class:`CatchUpProvider` for the events that bring its profiles up to date;
the engine runs the strategies that keep those profiles on them, and the
trigger returns once every run has settled (``status: ok``), or at its
timeout (``status: timeout``, the runs still going).

Each kind of profile kept on a cadence registers one provider in
``everos.memory.strategies.CATCH_UP_PROVIDERS``:

- :data:`USER_PROFILES` returns one :class:`~everos.memory.events.UserProfileDue`
  per user of the root whose own profile has segments queued:
  ``extract_user_profile`` reads them all.
- :data:`GROUP_PROFILES` returns one
  :class:`~everos.memory.events.GroupProfilesDue` per group of the root: its
  member profiles read every segment queued for them, the group profile its
  batch, a batch waiting out a back-off included.

A segment whose episode event is still on its way when the catch-up reads the
queues is queued after it: an evaluation triggers the catch-up once the
engine has settled (a second trigger reads what a first one missed).
"""

from __future__ import annotations

__all__ = ["GROUP_PROFILES", "USER_PROFILES", "CatchUpProvider", "catch_up_spec"]

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import GroupProfilesDue, UserProfileDue
from everos.memory.ports import Event, Immediate, ManualTick, Scope, StrategySpec

logger = get_logger(__name__)


@dataclass(frozen=True)
class CatchUpProvider:
    """One kind of profile a catch-up brings up to date.

    Attributes:
        name: The kind, as the catch-up logs it (``group_profiles``).
        due: The events that bring every profile of the kind up to date now;
            the strategies keeping them subscribe to those events.
        emits: The event classes ``due`` returns.
    """

    name: str
    due: Callable[[MemoryDeps], Awaitable[list[Event]]]
    emits: frozenset[type[Event]]


async def _groups_due(deps: MemoryDeps) -> list[Event]:
    """One :class:`GroupProfilesDue` per group of the root."""
    return [
        GroupProfilesDue(app_id=app_id, project_id=project_id, group_id=group_id)
        for app_id, project_id, group_id in await deps.stores.groups.group_keys()
    ]


GROUP_PROFILES = CatchUpProvider(
    name="group_profiles", due=_groups_due, emits=frozenset({GroupProfilesDue})
)
"""A group's member profiles and group profile (docs/group-memory.md, 7.4-7.5)."""


async def _users_due(deps: MemoryDeps) -> list[Event]:
    """One :class:`UserProfileDue` per user whose own profile has a queue."""
    due: list[Event] = []
    for app_id, project_id, user_id in await deps.stores.profile.user_keys():
        scope = Scope(
            app_id=app_id, project_id=project_id, owner_id=user_id, owner_type="user"
        )
        profile = await deps.stores.profile.read(scope)
        if profile is not None and profile.pending_memcell_ids:
            due.append(
                UserProfileDue(app_id=app_id, project_id=project_id, user_id=user_id)
            )
    return due


USER_PROFILES = CatchUpProvider(
    name="user_profiles", due=_users_due, emits=frozenset({UserProfileDue})
)
"""A user's own profile (``users/<u>/user.md``, ``extract_user_profile``)."""


def catch_up_spec(providers: Sequence[CatchUpProvider]) -> StrategySpec:
    """The ``catch_up_profiles`` strategy over ``providers``, in their order."""

    async def catch_up_profiles(event: ManualTick, deps: MemoryDeps) -> list[Event]:
        del event
        events: list[Event] = []
        for provider in providers:
            due = await provider.due(deps)
            logger.info("profiles_catch_up", provider=provider.name, events=len(due))
            events.extend(due)
        return events

    return StrategySpec(
        name="catch_up_profiles",
        fn=catch_up_profiles,
        trigger=Immediate(on=[ManualTick]),
        emits=frozenset().union(*(provider.emits for provider in providers)),
        max_retries=1,
    )
