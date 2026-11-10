"""The roster fold: events in, membership intervals out (docs/group-memory.md 4.3).

What these pin:

- join / leave / rejoin give one interval per stay, half-open ``[joined, left)``;
- a backdated join lands where its time says, not where it was appended;
- a leave without a join and a second open interval are conflicts (409), and a
  conflict says nothing was folded;
- ``update`` changes the descriptive fields from then on, and a rejoin keeps
  the member's last name and title unless the event sets new ones;
- windows merge adjacent intervals, which is what keeps a predicate short.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from everos.core.errors import ConflictError, RosterConflictError
from everos.memory.models import Roster, RosterEvent, fold_roster


def _join(member: str, at: int, **fields: object) -> RosterEvent:
    return RosterEvent(op="join", member_id=member, at=at, **fields)  # type: ignore[arg-type]


def _leave(member: str, at: int) -> RosterEvent:
    return RosterEvent(op="leave", member_id=member, at=at)


def test_join_leave_rejoin_gives_two_intervals() -> None:
    intervals = fold_roster(
        [
            _join("u_betty", 100, display_name="Betty"),
            _leave("u_betty", 200),
            _join("u_betty", 300),
        ]
    )

    assert [(i.seq, i.joined_at, i.left_at) for i in intervals] == [
        (1, 100, 200),
        (2, 300, None),
    ]
    # The rejoin keeps the name the member had.
    assert intervals[1].display_name == "Betty"
    roster = Roster(intervals=intervals)
    assert roster.is_current_member("u_betty")
    assert roster.windows_of("u_betty") == [(100, 200), (300, None)]
    assert roster.interval_at("u_betty", 250) is None
    assert roster.interval_at("u_betty", 199) is intervals[0]


def test_a_backdated_join_folds_at_its_own_time() -> None:
    """An importer replaying history appends a join that happened earlier."""
    intervals = fold_roster(
        [
            _join("u_chen", 500),
            _leave("u_chen", 900),
            _join("u_evan", 100),  # appended last, happened first
        ]
    )

    assert {(i.member_id, i.joined_at, i.left_at) for i in intervals} == {
        ("u_chen", 500, 900),
        ("u_evan", 100, None),
    }


@pytest.mark.parametrize(
    "events",
    [
        [_leave("u_betty", 100)],
        [_join("u_betty", 100), _join("u_betty", 200)],
        [RosterEvent(op="update", member_id="u_betty", at=100, title="PM")],
        [_join("u_betty", 100), _leave("u_betty", 200), _leave("u_betty", 300)],
    ],
    ids=["leave-without-join", "second-open", "update-non-member", "double-leave"],
)
def test_conflicts_are_409(events: list[RosterEvent]) -> None:
    with pytest.raises(RosterConflictError):
        fold_roster(events)
    assert issubclass(RosterConflictError, ConflictError)  # the 409 family


def test_update_changes_fields_and_kind_defaults_follow_role() -> None:
    intervals = fold_roster(
        [
            _join("bot_ava", 0, role="assistant", display_name="Ava"),
            _join("u_chen", 0, role="owner", title="Organiser"),
            RosterEvent(
                op="update", member_id="u_chen", at=50, aliases=["C", "Chen, Jr."]
            ),
        ]
    )
    by_member = {i.member_id: i for i in intervals}

    assert by_member["bot_ava"].kind == "assistant"
    assert by_member["bot_ava"].role == "assistant"
    assert by_member["u_chen"].kind == "human"
    assert by_member["u_chen"].title == "Organiser"
    assert by_member["u_chen"].aliases == ["C", "Chen, Jr."]
    assert Roster(intervals=intervals).has_managers()


def test_adjacent_intervals_merge_into_one_window() -> None:
    roster = Roster(
        intervals=fold_roster(
            [_join("m", 10), _leave("m", 20), _join("m", 20), _leave("m", 40)]
        )
    )

    assert roster.windows_of("m") == [(10, 40)]
    assert not roster.is_current_member("m")


_OPS = st.lists(
    st.tuples(
        st.sampled_from(["join", "leave"]),
        st.sampled_from(["a", "b"]),
        st.integers(min_value=0, max_value=50),
    ),
    max_size=12,
)


@settings(max_examples=150, deadline=None)
@given(_OPS)
def test_a_fold_either_conflicts_or_gives_disjoint_ordered_intervals(
    ops: list[tuple[str, str, int]],
) -> None:
    """Whatever the event soup: intervals of one member never overlap."""
    events = [RosterEvent(op=op, member_id=m, at=t) for op, m, t in ops]  # type: ignore[arg-type]
    try:
        intervals = fold_roster(events)
    except RosterConflictError:
        return
    for member in ("a", "b"):
        mine = sorted(
            (i for i in intervals if i.member_id == member), key=lambda i: i.seq
        )
        assert [i.seq for i in mine] == list(range(1, len(mine) + 1))
        for earlier, later in itertools.pairwise(mine):
            assert earlier.left_at is not None
            assert earlier.left_at <= later.joined_at
        assert sum(1 for i in mine if i.left_at is None) <= 1
