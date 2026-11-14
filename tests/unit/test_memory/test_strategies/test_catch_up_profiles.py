"""catch_up_profiles: every profile kept on a cadence, brought up to date at once.

What these pin:

- the strategy runs on a manual trigger only and returns what its providers
  ask for: one ``GroupProfilesDue`` per group of the root, every space;
- the providers are a registry: ``CATCH_UP_PROVIDERS`` lists the groups'
  provider, and the spec's declared events are the union of its providers';
- a group's two profile strategies read their batches on ``GroupProfilesDue``.
"""

from __future__ import annotations

from typing import Any

from everos.memory.events import Event, GroupProfilesDue
from everos.memory.models import Group
from everos.memory.ports import Immediate, ManualTick
from everos.memory.strategies import (
    CATCH_UP_PROFILES_SPEC,
    CATCH_UP_PROVIDERS,
    GROUP_PROFILES,
    UPDATE_GROUP_PROFILE_SPEC,
    UPDATE_MEMBER_PROFILES_SPEC,
    CatchUpProvider,
    catch_up_spec,
)
from tests.fakes import make_deps

_TICK = ManualTick(strategy_name="catch_up_profiles")


async def test_one_event_per_group_of_every_space() -> None:
    deps = make_deps()
    for key in (("a", "p", "trip"), ("a", "q", "ops"), ("b", "p", "trip")):
        deps.stores.groups.groups[key] = Group(group_id=key[2], created_at=0)

    events = await CATCH_UP_PROFILES_SPEC.fn(_TICK, deps)

    assert all(isinstance(e, GroupProfilesDue) for e in events)
    assert [(e.app_id, e.project_id, e.group_id) for e in events] == [
        ("a", "p", "trip"),
        ("a", "q", "ops"),
        ("b", "p", "trip"),
    ]


async def test_it_runs_on_a_manual_trigger_only() -> None:
    trigger = CATCH_UP_PROFILES_SPEC.trigger
    assert isinstance(trigger, Immediate) and trigger.on == [ManualTick]
    assert CATCH_UP_PROVIDERS == (GROUP_PROFILES,)
    assert CATCH_UP_PROFILES_SPEC.emits == frozenset({GroupProfilesDue})


async def test_a_registered_provider_adds_its_events() -> None:
    class Done(Event):
        owner: str

    async def due(deps: Any) -> list[Event]:
        return [Done(owner="u_ann")]

    other = CatchUpProvider(name="users", due=due, emits=frozenset({Done}))
    spec = catch_up_spec((GROUP_PROFILES, other))

    assert spec.emits == frozenset({GroupProfilesDue, Done})
    (done,) = await spec.fn(_TICK, make_deps())
    assert isinstance(done, Done) and done.owner == "u_ann"


def test_both_group_profile_strategies_read_on_the_catch_up() -> None:
    for spec in (UPDATE_MEMBER_PROFILES_SPEC, UPDATE_GROUP_PROFILE_SPEC):
        assert isinstance(spec.trigger, Immediate)
        assert GroupProfilesDue in spec.trigger.on
