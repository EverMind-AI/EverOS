"""The group's deliverer in the access policy: one rule, every right of an owner.

What these pin (docs/group-memory.md, sections 5.1 and 9.1):

- the deliverer passes every ``authorize_*`` check an owner passes -- meta,
  any roster event (backdated too), edit and forget of anyone's records --
  without being a member, and a stranger still passes none;
- ``authorize_delivery``: anyone delivers to a group without a deliverer;
  once it has one, only the deliverer, with a clear 403 for anyone else;
- a read scope gives the deliverer the whole history; a search that names no
  audience (``ScopeRequest.audience = None``) reads as ``group`` -- only the
  time every current member can see -- and ``asker`` gives the full view;
- the deliverer is never one of the ``audience = group`` intersection, even
  when it is a member too, and a member's windows are what they were;
- a group the asker does not deliver to, and is not a member of, is 403.
"""

from __future__ import annotations

import pytest

from everos.core.errors import ForbiddenError
from everos.infra.persistence.groups import PartitionProjectLinkStore
from everos.memory.access import (
    AccessPolicy,
    Identity,
    ScopeRequest,
    Window,
    admits,
    authorize_delivery,
    authorize_edit,
    authorize_forget,
    authorize_manage,
    authorize_roster_change,
    is_deliverer,
)
from everos.memory.models import Group, Roster, RosterEvent, fold_roster
from everos.memory.ports import Scope
from tests.fakes import InMemoryGroupStore

_APP, _PROJECT = "default", "p"
_BOT = "gw_trip"


def _join(member: str, at: int, **fields: object) -> RosterEvent:
    return RosterEvent(op="join", member_id=member, at=at, **fields)  # type: ignore[arg-type]


def _leave(member: str, at: int) -> RosterEvent:
    return RosterEvent(op="leave", member_id=member, at=at)


_EVENTS = [
    _join("u_chen", 0, role="owner"),
    _join("u_betty", 0),
    _join("u_evan", 5000),
    _join("u_dan", 0),
    _leave("u_dan", 2500),
]
"""Chen owns the group; Betty is a member from the start; Evan joined late;
Dan left."""


def _roster(deliverer: str | None = _BOT) -> Roster:
    return Roster(intervals=fold_roster(_EVENTS), deliverer_id=deliverer)


def _scope(group_id: str) -> Scope:
    return Scope(
        app_id=_APP, project_id=_PROJECT, owner_id=group_id, owner_type="group"
    )


async def _policy(*, deliverer: str | None = _BOT) -> AccessPolicy:
    store = InMemoryGroupStore()
    await store.save(
        _scope("trip"),
        group=Group(group_id="trip", created_at=0, deliverer_id=deliverer),
        events=_EVENTS,
    )
    await store.save(
        _scope("other"),
        group=Group(group_id="other", created_at=0, deliverer_id="gw_other"),
        events=[_join("u_zoe", 0, role="owner")],
    )
    return AccessPolicy(store, projects=PartitionProjectLinkStore(store))


def _windows(scope_windows: tuple[Window, ...]) -> list[tuple[int | None, int | None]]:
    return [(w.start, w.end) for w in scope_windows]


def test_the_deliverer_holds_every_right_of_an_owner() -> None:
    roster = _roster()
    assert is_deliverer(roster, _BOT)
    assert admits(roster, _BOT) and not roster.is_current_member(_BOT)
    authorize_manage(roster, _BOT)
    # Any event, backdated before the newest one in the roster too.
    authorize_roster_change(
        roster, _BOT, [_join("u_fay", 100), _leave("u_betty", 6000)]
    )
    authorize_edit(roster, _BOT, sources=["u_betty"])
    authorize_forget(roster, _BOT, sources=["u_betty", "u_evan"])
    authorize_forget(roster, _BOT, sources=[])  # a whole session


def test_a_stranger_and_a_plain_member_keep_today_s_answers() -> None:
    roster = _roster()
    assert not admits(roster, "u_zoe") and not admits(roster, None)
    for check in (
        lambda: authorize_manage(roster, "u_zoe"),
        lambda: authorize_manage(roster, "u_betty"),
        lambda: authorize_roster_change(roster, "u_betty", [_join("u_fay", 100)]),
        lambda: authorize_edit(roster, "u_betty", sources=["u_chen"]),
        lambda: authorize_forget(roster, "u_betty", sources=["u_chen"]),
        lambda: authorize_forget(roster, None, sources=["u_chen"]),
    ):
        with pytest.raises(ForbiddenError):
            check()
    authorize_forget(roster, "u_betty", sources=["u_betty"])
    authorize_manage(roster, "u_chen")


def test_only_the_deliverer_delivers_once_the_group_has_one() -> None:
    for anyone in ("u_chen", "u_zoe", None):
        authorize_delivery(_roster(deliverer=None), anyone, group_id="trip")
    authorize_delivery(_roster(), _BOT, group_id="trip")
    for other in ("u_chen", "u_betty", "gw_other", None):
        with pytest.raises(ForbiddenError, match="only the deliverer of group 'trip'"):
            authorize_delivery(_roster(), other, group_id="trip")


async def test_the_deliverer_reads_the_whole_history_and_searches_as_the_group() -> (
    None
):
    policy = await _policy()
    full = await policy.read_scope(
        Identity(_BOT), ScopeRequest(project_id=_PROJECT, group_ids=("trip",))
    )
    assert full.audience == "asker"
    assert _windows(full.groups[0].windows) == [(None, None)]

    # A search that names no audience: only what every current member sees --
    # Evan's window, the latest join.
    default = await policy.read_scope(
        Identity(_BOT),
        ScopeRequest(project_id=_PROJECT, group_ids=("trip",), audience=None),
    )
    assert default.audience == "group"
    assert _windows(default.groups[0].windows) == [(5000, None)]

    explicit = await policy.read_scope(
        Identity(_BOT),
        ScopeRequest(project_id=_PROJECT, group_ids=("trip",), audience="group"),
    )
    assert _windows(explicit.groups[0].windows) == [(5000, None)]


async def test_a_member_s_view_is_unchanged_by_a_deliverer() -> None:
    for deliverer in (None, _BOT):
        policy = await _policy(deliverer=deliverer)
        for asker, expected in (("u_chen", [(0, None)]), ("u_evan", [(5000, None)])):
            for audience in ("asker", None):
                scope = await policy.read_scope(
                    Identity(asker),
                    ScopeRequest(
                        project_id=_PROJECT, group_ids=("trip",), audience=audience
                    ),
                )
                assert scope.audience == "asker"
                assert _windows(scope.groups[0].windows) == expected


async def test_the_deliverer_is_never_one_of_the_audience() -> None:
    # The deliverer is also the group's assistant, joined late: a member's
    # reply to the whole group is not narrowed by the deliverer's window.
    store = InMemoryGroupStore()
    await store.save(
        _scope("trip"),
        group=Group(group_id="trip", created_at=0, deliverer_id=_BOT),
        events=[*_EVENTS, _join(_BOT, 7000, kind="assistant", role="assistant")],
    )
    policy = AccessPolicy(store, projects=PartitionProjectLinkStore(store))
    for asker, expected in (("u_chen", [(5000, None)]), (_BOT, [(5000, None)])):
        scope = await policy.read_scope(
            Identity(asker),
            ScopeRequest(project_id=_PROJECT, group_ids=("trip",), audience="group"),
        )
        assert _windows(scope.groups[0].windows) == expected
    own = await policy.read_scope(
        Identity(_BOT), ScopeRequest(project_id=_PROJECT, group_ids=("trip",))
    )
    assert _windows(own.groups[0].windows) == [(None, None)]


async def test_the_deliverer_reads_nothing_of_another_group() -> None:
    policy = await _policy()
    for groups in (("other",), ("trip", "other")):
        with pytest.raises(ForbiddenError):
            await policy.read_scope(
                Identity(_BOT), ScopeRequest(project_id=_PROJECT, group_ids=groups)
            )
    with pytest.raises(ForbiddenError):
        await policy.read_scope(
            Identity("gw_other"),
            ScopeRequest(project_id=_PROJECT, group_ids=("trip",)),
        )
