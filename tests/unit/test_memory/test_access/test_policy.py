"""The access policy: one predicate per read, compiled from the roster.

What these pin (docs/group-memory.md, sections 9 and 14 task 3):

- property: over random rosters and record times, a group record matches the
  compiled predicate iff the asker is a current member and the record's whole
  source span lies inside one of the asker's windows;
- a former member, a non-member and an unknown group are all 403, with no
  difference between "exists" and "does not";
- a personal read by another asker, and a header that disagrees with the
  body's ``asker_id``, are 403; a personal read without an asker is today's;
- ``history_visible_from = creation`` and ``audience = group`` change the
  windows exactly as section 15 says;
- the user predicate is the one recalls used before the policy existed;
- a project scope and ``include_groups`` read the groups the project links
  answer (section 4.4), not the space's; the rosters still decide what the
  asker sees, and a linked group of another space is refused.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from everos.core.errors import ForbiddenError, InvalidInputError
from everos.core.utils.datetime import from_timestamp_ms
from everos.infra.persistence.groups import PartitionProjectLinkStore
from everos.memory.access import (
    AccessPolicy,
    Identity,
    ScopeRequest,
    Window,
    intersect_windows,
    owner_predicate,
    resolve_identity,
    visible,
)
from everos.memory.models import CanonicalMessage, Group, RosterEvent
from everos.memory.ports import Scope, SessionKey, all_of, eq, is_null
from tests.fakes import InMemoryGroupStore
from tests.fakes._filter import matches

_APP, _PROJECT = "default", "p"


def _scope(group_id: str) -> Scope:
    return Scope(
        app_id=_APP, project_id=_PROJECT, owner_id=group_id, owner_type="group"
    )


async def _store(
    events: list[RosterEvent], *, group_id: str = "g", visible_from: str = "join"
) -> InMemoryGroupStore:
    store = InMemoryGroupStore()
    await store.save(
        _scope(group_id),
        group=Group(
            group_id=group_id,
            created_at=0,
            history_visible_from=visible_from,  # type: ignore[arg-type]
        ),
        events=events,
    )
    return store


def _policy(store: InMemoryGroupStore) -> AccessPolicy:
    """The policy over ``store``, a project's groups by the partition rule."""
    return AccessPolicy(store, projects=PartitionProjectLinkStore(store))


def _row(group_id: str, start: int, end: int, **extra: Any) -> dict[str, Any]:
    return {
        "owner_type": "group",
        "owner_id": group_id,
        "app_id": _APP,
        "project_id": _PROJECT,
        "deprecated_by": None,
        "source_start": from_timestamp_ms(start),
        "source_end": from_timestamp_ms(end),
        **extra,
    }


def _request(*group_ids: str, audience: str = "asker") -> ScopeRequest:
    return ScopeRequest(
        app_id=_APP,
        project_id=_PROJECT,
        group_ids=tuple(group_ids),
        audience=audience,  # type: ignore[arg-type]
    )


# ── the property ─────────────────────────────────────────────────────────

_MEMBERS = ("a", "b", "c")


@st.composite
def _rosters(draw: st.DrawFn) -> list[RosterEvent]:
    """A valid event log: per member, alternating joins and leaves."""
    events: list[RosterEvent] = []
    for member in _MEMBERS:
        times = sorted(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=100),
                    unique=True,
                    max_size=5,
                )
            )
        )
        for i, at in enumerate(times):
            op = "join" if i % 2 == 0 else "leave"
            events.append(RosterEvent(op=op, member_id=member, at=at))  # type: ignore[arg-type]
    return events


def _oracle(events: list[RosterEvent], asker: str, start: int, end: int) -> bool:
    """Continuously a member over ``[start, end]``, and a member now."""
    spans: list[tuple[int, int | None]] = []
    for event in sorted(
        (e for e in events if e.member_id == asker), key=lambda e: e.at
    ):
        if event.op == "join":
            spans.append((event.at, None))
        else:
            spans[-1] = (spans[-1][0], event.at)
    if not spans or spans[-1][1] is not None:
        return False  # not a current member: no read at all
    merged: list[tuple[int, int | None]] = []
    for joined, left in spans:
        if merged and merged[-1][1] == joined:
            merged[-1] = (merged[-1][0], left)
        else:
            merged.append((joined, left))
    return any(
        start >= joined and (left is None or end < left) for joined, left in merged
    )


@settings(max_examples=200, deadline=None)
@given(
    events=_rosters(),
    asker=st.sampled_from(_MEMBERS),
    span=st.tuples(
        st.integers(min_value=0, max_value=110), st.integers(min_value=0, max_value=20)
    ),
)
async def test_a_record_is_visible_iff_its_sources_lie_in_a_window(
    events: list[RosterEvent], asker: str, span: tuple[int, int]
) -> None:
    store = await _store(events)
    policy = _policy(store)
    start, length = span
    end = start + length
    expected = _oracle(events, asker, start, end)
    try:
        scope = await policy.read_scope(Identity(asker_id=asker), _request("g"))
    except ForbiddenError:
        assert not expected
        return
    record = _row("g", start, end)
    assert matches(scope.predicate, record) is expected
    assert (
        visible(
            scope,
            owner_type="group",
            owner_id="g",
            source_start=start,
            source_end=end,
        )
        is expected
    )
    # Another group's record never matches, whatever its times.
    assert not matches(scope.predicate, _row("other", start, end))


# ── membership and identity ──────────────────────────────────────────────


async def test_former_member_non_member_and_unknown_group_are_alike() -> None:
    store = await _store(
        [
            RosterEvent(op="join", member_id="u_betty", at=10),
            RosterEvent(op="leave", member_id="u_betty", at=20),
            RosterEvent(op="join", member_id="u_chen", at=10),
        ]
    )
    policy = _policy(store)

    errors = []
    for asker, group in (("u_betty", "g"), ("u_zoe", "g"), ("u_chen", "nope")):
        with pytest.raises(ForbiddenError) as caught:
            await policy.read_scope(Identity(asker_id=asker), _request(group))
        errors.append(str(caught.value).replace(repr(group), "<g>"))
    assert len(set(errors)) == 1


async def test_a_group_scope_needs_an_asker() -> None:
    policy = _policy(await _store([RosterEvent(op="join", member_id="a", at=1)]))
    with pytest.raises(InvalidInputError):
        await policy.read_scope(Identity(), _request("g"))


async def test_a_personal_scope_is_read_by_its_owner_only() -> None:
    policy = _policy(InMemoryGroupStore())
    request = ScopeRequest(app_id=_APP, project_id=_PROJECT, user_id="u_betty")

    with pytest.raises(ForbiddenError):
        await policy.read_scope(Identity(asker_id="u_chen"), request)
    own = await policy.read_scope(Identity(asker_id="u_betty"), request)
    anonymous = await policy.read_scope(Identity(), request)
    assert own.predicate == anonymous.predicate
    assert anonymous.asker_id == "u_betty"


def test_a_header_that_disagrees_with_the_body_is_403() -> None:
    with pytest.raises(ForbiddenError):
        resolve_identity(header="u_chen", body_asker="u_betty")
    assert resolve_identity(header="u_chen", body_asker=None).asker_id == "u_chen"
    assert resolve_identity(header=None, body_asker="u_betty").asker_id == "u_betty"
    assert resolve_identity(header="", body_asker=None).asker_id is None


def test_the_user_predicate_is_the_one_recalls_always_used() -> None:
    """Byte-identical one-to-one queries depend on this tree, node for node."""
    assert owner_predicate(
        owner_id="u1", owner_type="user", app_id="a", project_id="p"
    ) == all_of(
        eq("owner_id", "u1"),
        eq("owner_type", "user"),
        eq("app_id", "a"),
        eq("project_id", "p"),
        is_null("deprecated_by"),
    )


# ── the two settings of section 15 ───────────────────────────────────────


async def test_history_visible_from_creation_shows_a_late_joiner_everything() -> None:
    events = [
        RosterEvent(op="join", member_id="u_chen", at=10),
        RosterEvent(op="join", member_id="u_evan", at=500),
    ]
    strict = _policy(await _store(events))
    open_history = _policy(await _store(events, visible_from="creation"))
    early = _row("g", 20, 30)

    strict_scope = await strict.read_scope(Identity(asker_id="u_evan"), _request("g"))
    open_scope = await open_history.read_scope(
        Identity(asker_id="u_evan"), _request("g")
    )
    assert not matches(strict_scope.predicate, early)
    assert matches(open_scope.predicate, early)


async def test_audience_group_keeps_what_every_current_member_can_see() -> None:
    store = await _store(
        [
            RosterEvent(op="join", member_id="u_chen", at=10),
            RosterEvent(op="join", member_id="u_betty", at=10),
            RosterEvent(op="join", member_id="u_evan", at=500),
        ]
    )
    policy = _policy(store)
    early, late = _row("g", 20, 30), _row("g", 600, 610)

    mine = await policy.read_scope(Identity(asker_id="u_chen"), _request("g"))
    everyone = await policy.read_scope(
        Identity(asker_id="u_chen"), _request("g", audience="group")
    )
    assert matches(mine.predicate, early) and matches(mine.predicate, late)
    assert not matches(everyone.predicate, early)
    assert matches(everyone.predicate, late)


def test_windows_intersect_as_time_ranges() -> None:
    assert intersect_windows([Window(0, 10), Window(20, None)], [Window(5, 25)]) == (
        Window(5, 10),
        Window(20, 25),
    )
    assert intersect_windows([Window(None, None)], [Window(3, None)]) == (
        Window(3, None),
    )
    assert intersect_windows([Window(0, 5)], [Window(5, 9)]) == ()


async def test_several_groups_form_one_pool_each_with_its_windows() -> None:
    store = await _store([RosterEvent(op="join", member_id="u_chen", at=100)])
    await store.save(
        _scope("h"),
        group=Group(group_id="h", created_at=0),
        events=[RosterEvent(op="join", member_id="u_chen", at=0)],
    )
    scope = await _policy(store).read_scope(
        Identity(asker_id="u_chen"), _request("g", "h")
    )

    assert scope.owner_ids == ("g", "h")
    assert not matches(scope.predicate, _row("g", 50, 60))  # before joining g
    assert matches(scope.predicate, _row("h", 50, 60))
    assert matches(scope.predicate, _row("g", 150, 160))


# ── a personal read that covers its owner's groups (include_groups) ─────


def _personal(user: str) -> ScopeRequest:
    return ScopeRequest(app_id=_APP, project_id=_PROJECT, user_id=user)


@settings(max_examples=100, deadline=None)
@given(events=_rosters(), owner=st.sampled_from(_MEMBERS))
async def test_member_groups_compiles_each_group_as_the_owners_group_read(
    events: list[RosterEvent], owner: str
) -> None:
    """The same scope, predicate for predicate, as a group read by the owner;
    nothing for a group the owner is not a current member of."""
    policy = _policy(await _store(events))
    personal = await policy.read_scope(Identity(), _personal(owner))
    covered = await policy.member_groups(personal)
    try:
        group = await policy.read_scope(Identity(asker_id=owner), _request("g"))
    except ForbiddenError:
        assert covered is None
        return
    assert covered is not None
    assert covered.predicate == group.predicate
    assert covered.history_predicate == group.history_predicate
    assert covered.groups == group.groups
    assert (covered.asker_id, covered.audience) == (owner, "asker")


async def test_member_groups_takes_the_current_groups_of_the_space_only() -> None:
    store = await _store([RosterEvent(op="join", member_id="u_chen", at=100)])
    for group_id, events in (
        ("h", [RosterEvent(op="join", member_id="u_chen", at=0)]),
        (
            "left",
            [
                RosterEvent(op="join", member_id="u_chen", at=0),
                RosterEvent(op="leave", member_id="u_chen", at=50),
            ],
        ),
        ("theirs", [RosterEvent(op="join", member_id="u_zoe", at=0)]),
    ):
        await store.save(
            _scope(group_id),
            group=Group(group_id=group_id, created_at=0),
            events=events,
        )
    await store.save(
        Scope(app_id=_APP, project_id="other", owner_id="far", owner_type="group"),
        group=Group(group_id="far", created_at=0),
        events=[RosterEvent(op="join", member_id="u_chen", at=0)],
    )
    policy = _policy(store)
    personal = await policy.read_scope(Identity(asker_id="u_chen"), _personal("u_chen"))

    every = await policy.member_groups(personal)
    listed = await policy.member_groups(personal, only=["h", "left", "nowhere"])
    none = await policy.member_groups(personal, only=["left", "theirs", "far"])

    assert every is not None and every.owner_ids == ("g", "h")
    assert listed is not None and listed.owner_ids == ("h",)
    assert none is None
    assert not matches(every.predicate, _row("left", 10, 20))
    assert not matches(every.predicate, _row("theirs", 10, 20))
    assert matches(every.predicate, _row("h", 10, 20))
    assert not matches(every.predicate, _row("g", 10, 20))  # before joining g


async def test_only_a_personal_read_covers_its_owners_groups() -> None:
    policy = _policy(await _store([RosterEvent(op="join", member_id="a", at=1)]))
    group = await policy.read_scope(Identity(asker_id="a"), _request("g"))
    with pytest.raises(ValueError, match="personal read"):
        await policy.member_groups(group)


# ── a project's groups: the project links' answer (section 4.4) ─────────


class _Links:
    """A ``ProjectLinkStore`` that links exactly the groups it is given."""

    def __init__(self, *groups: Scope) -> None:
        self.groups = list(groups)
        self.asked: list[tuple[str, str]] = []

    async def groups_of(self, app_id: str, project_id: str) -> list[Scope]:
        self.asked.append((app_id, project_id))
        return list(self.groups)


_PROJECT_SCOPE = ScopeRequest(app_id=_APP, project_id=_PROJECT, project_scope=True)


async def _three_groups() -> InMemoryGroupStore:
    """``g``, ``h`` and ``k`` in the space, Chen a current member of each."""
    store = await _store([RosterEvent(op="join", member_id="u_chen", at=0)])
    for group_id in ("h", "k"):
        await store.save(
            _scope(group_id),
            group=Group(group_id=group_id, created_at=0),
            events=[RosterEvent(op="join", member_id="u_chen", at=0)],
        )
    return store


async def test_a_project_reads_its_linked_groups_not_its_space() -> None:
    """The links name the groups; a group of the space they leave out is not read."""
    links = _Links(_scope("h"), _scope("k"))
    policy = AccessPolicy(await _three_groups(), projects=links)
    chen = Identity(asker_id="u_chen")

    scope = await policy.read_scope(chen, _PROJECT_SCOPE)
    covered = await policy.member_groups(
        await policy.read_scope(chen, _personal("u_chen"))
    )

    assert scope.owner_ids == ("h", "k")
    assert covered is not None and covered.owner_ids == ("h", "k")
    assert links.asked == [(_APP, _PROJECT), (_APP, _PROJECT)]
    assert matches(scope.predicate, _row("h", 10, 20))
    assert not matches(scope.predicate, _row("g", 10, 20))


async def test_the_rosters_still_decide_what_a_project_shows() -> None:
    """A linked group the asker is not in adds nothing; none at all is 403."""
    store = await _three_groups()
    await store.save(
        _scope("theirs"),
        group=Group(group_id="theirs", created_at=0),
        events=[RosterEvent(op="join", member_id="u_zoe", at=0)],
    )
    chen = Identity(asker_id="u_chen")

    scope = await AccessPolicy(
        store, projects=_Links(_scope("theirs"), _scope("h"))
    ).read_scope(chen, _PROJECT_SCOPE)

    assert scope.owner_ids == ("h",)
    for links in (_Links(_scope("theirs")), _Links()):
        with pytest.raises(ForbiddenError):
            await AccessPolicy(store, projects=links).read_scope(chen, _PROJECT_SCOPE)


async def test_a_linked_group_of_another_space_is_refused_not_read_empty() -> None:
    """A read scope covers one space; a link out of it is an error."""
    store = await _three_groups()
    far = Scope(app_id=_APP, project_id="other", owner_id="far", owner_type="group")
    await store.save(
        far,
        group=Group(group_id="far", created_at=0),
        events=[RosterEvent(op="join", member_id="u_chen", at=0)],
    )
    policy = AccessPolicy(store, projects=_Links(_scope("h"), far))
    chen = Identity(asker_id="u_chen")
    personal = await policy.read_scope(chen, _personal("u_chen"))

    with pytest.raises(ValueError, match="own space"):
        await policy.read_scope(chen, _PROJECT_SCOPE)
    with pytest.raises(ValueError, match="own space"):
        await policy.member_groups(personal)


# ── buffered messages (row 3) ────────────────────────────────────────────


def _message(sender: str, at_ms: int) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=f"m{at_ms}",
        session_id="s",
        sender_id=sender,
        role="user",
        timestamp=from_timestamp_ms(at_ms),
        text="hi",
    )


async def _no_memory() -> bool:
    return False


async def test_a_group_buffer_is_cut_to_the_asker_window() -> None:
    store = await _store(
        [
            RosterEvent(op="join", member_id="u_chen", at=0),
            RosterEvent(op="join", member_id="u_evan", at=5_000),
        ]
    )
    session = SessionKey(app_id=_APP, project_id=_PROJECT, session_id="s")
    await store.bind(session, "g", at=0)
    policy = _policy(store)
    buffered = [_message("u_chen", 1_000), _message("u_chen", 6_000)]

    evan = await policy.read_scope(Identity(asker_id="u_evan"), _request("g"))
    seen = await policy.buffered_messages(
        evan, session, buffered, owner_has_memory=_no_memory
    )
    assert [m.message_id for m in seen] == ["m6000"]

    # The same session read through a personal scope shows nothing at all.
    chen_personal = await policy.read_scope(
        Identity(), ScopeRequest(app_id=_APP, project_id=_PROJECT, user_id="u_chen")
    )
    assert (
        await policy.buffered_messages(
            chen_personal, session, buffered, owner_has_memory=_no_memory
        )
        == []
    )


async def test_a_one_to_one_buffer_needs_a_sender_of_the_session() -> None:
    policy = _policy(InMemoryGroupStore())
    session = SessionKey(app_id=_APP, project_id=_PROJECT, session_id="s")
    buffered = [_message("u_alice", 1_000)]

    async def has_memory() -> bool:
        return True

    def personal(user: str) -> ScopeRequest:
        return ScopeRequest(app_id=_APP, project_id=_PROJECT, user_id=user)

    alice = await policy.read_scope(Identity(), personal("u_alice"))
    mallory = await policy.read_scope(Identity(), personal("u_mallory"))
    assert (
        await policy.buffered_messages(
            alice, session, buffered, owner_has_memory=_no_memory
        )
        == buffered
    )
    assert (
        await policy.buffered_messages(
            mallory, session, buffered, owner_has_memory=_no_memory
        )
        == []
    )
    # Took part earlier (memory from the session), nothing buffered of theirs.
    assert (
        await policy.buffered_messages(
            mallory, session, buffered, owner_has_memory=has_memory
        )
        == buffered
    )
