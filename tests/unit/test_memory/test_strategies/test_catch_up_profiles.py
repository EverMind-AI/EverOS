"""catch_up_profiles: every profile kept on a cadence, brought up to date at once.

What these pin:

- the strategy runs on a manual trigger only and returns what its providers
  ask for: one ``UserProfileDue`` per user of the root whose own profile has
  segments queued, then one ``GroupProfilesDue`` per group of the root, every
  space;
- the providers are a registry: ``CATCH_UP_PROVIDERS`` lists the users' and
  the groups' providers, and the spec's declared events are the union of its
  providers';
- a group's two profile strategies read their batches on ``GroupProfilesDue``,
  and the one-to-one profile reads its queue on ``UserProfileDue``.
"""

from __future__ import annotations

from typing import Any

from everos.memory.events import Event, GroupProfilesDue, UserProfileDue
from everos.memory.models import ExplicitInfoItem, Group, Profile
from everos.memory.ports import Immediate, ManualTick, Scope
from everos.memory.strategies import (
    CATCH_UP_PROFILES_SPEC,
    CATCH_UP_PROVIDERS,
    EXTRACT_USER_PROFILE_SPEC,
    GROUP_PROFILES,
    UPDATE_GROUP_PROFILE_SPEC,
    UPDATE_MEMBER_PROFILES_SPEC,
    USER_PROFILES,
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


def _user(app_id: str, user_id: str, project_id: str = "p") -> Scope:
    return Scope(
        app_id=app_id, project_id=project_id, owner_id=user_id, owner_type="user"
    )


def _profile(owner_id: str, *pending: str) -> Profile:
    return Profile(
        owner_id=owner_id,
        summary="s",
        timestamp=0,
        explicit_info=[ExplicitInfoItem(description="Lives in Lyon.")],
        pending_memcell_ids=list(pending),
    )


async def test_one_event_per_user_with_a_queue_then_one_per_group() -> None:
    deps = make_deps()
    await deps.stores.profile.write(_user("b", "u_cid"), _profile("u_cid", "mc_9"))
    await deps.stores.profile.write(_user("a", "u_bob"), _profile("u_bob"))
    await deps.stores.profile.write(
        _user("a", "u_ann"), _profile("u_ann", "mc_1", "mc_2")
    )
    # A member's profile in a group with a queue is the group's provider's.
    group = Scope(app_id="a", project_id="p", owner_id="trip", owner_type="group")
    await deps.stores.profile.write(group, _profile("u_ann", "mc_3"))
    deps.stores.groups.groups[("a", "p", "trip")] = Group(group_id="trip", created_at=0)

    events = await CATCH_UP_PROFILES_SPEC.fn(_TICK, deps)

    users = [e for e in events if isinstance(e, UserProfileDue)]
    assert [(e.app_id, e.project_id, e.user_id) for e in users] == [
        ("a", "p", "u_ann"),
        ("b", "p", "u_cid"),
    ]
    assert events[: len(users)] == users  # the users' provider runs first
    (due,) = events[len(users) :]
    assert isinstance(due, GroupProfilesDue) and due.group_id == "trip"


async def test_it_runs_on_a_manual_trigger_only() -> None:
    trigger = CATCH_UP_PROFILES_SPEC.trigger
    assert isinstance(trigger, Immediate) and trigger.on == [ManualTick]
    assert CATCH_UP_PROVIDERS == (USER_PROFILES, GROUP_PROFILES)
    assert CATCH_UP_PROFILES_SPEC.emits == frozenset({UserProfileDue, GroupProfilesDue})


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


def test_the_user_profile_reads_on_the_catch_up() -> None:
    due = UserProfileDue(user_id="u_ann", app_id="a", project_id="p")
    assert isinstance(EXTRACT_USER_PROFILE_SPEC.trigger, Immediate)
    assert UserProfileDue in EXTRACT_USER_PROFILE_SPEC.trigger.on
    assert EXTRACT_USER_PROFILE_SPEC.applies_to is not None
    assert EXTRACT_USER_PROFILE_SPEC.applies_to(due, make_deps())
