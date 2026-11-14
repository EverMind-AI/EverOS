"""catch_up_profiles strategy -- every profile on a cadence, up to date at once.

The profiles follow a cadence, not every segment: a group's member profiles
and group profile read their batches after ``[groups] group_profile_batch``
new segments or ``group_profile_idle_seconds`` without one, and a flush no
longer reads them unless ``[groups] profile_on_flush`` is set. Whoever needs
them current at once -- an evaluation about to search what it just
ingested, an operator -- triggers this strategy::

    POST /api/v2/ome/trigger  {"name": "catch_up_profiles", "timeout": 1800}

It runs on that :class:`~everos.memory.ports.ManualTick` only and asks every
:class:`CatchUpProvider` for the events that bring its profiles up to date;
the engine runs the strategies that keep those profiles on them, and the
trigger returns once every run has settled (``status: ok``), or at its
timeout (``status: timeout``, the runs still going).

Each kind of profile kept on a cadence registers one provider in
``everos.memory.strategies.CATCH_UP_PROVIDERS``. :data:`GROUP_PROFILES`, the
groups' provider, returns one :class:`~everos.memory.events.GroupProfilesDue`
per group of the root: its member profiles read every segment queued for
them, the group profile its batch, a batch waiting out a back-off included.
A one-to-one profile with a cadence of its own registers its provider beside
it.
"""

from __future__ import annotations

__all__ = ["GROUP_PROFILES", "CatchUpProvider", "catch_up_spec"]

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import GroupProfilesDue
from everos.memory.ports import Event, Immediate, ManualTick, StrategySpec

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
