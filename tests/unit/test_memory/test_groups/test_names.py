"""Name matching for group attribution (docs/group-memory.md, sections 7.2, 8.2).

What these pin:

- a name matches as a whole word, case-insensitively, in a script with word
  spacing (``Al`` is not found inside ``Alice``);
- in a script written without word spacing a name matches as a substring;
- aliases count like display names; results follow the order of the text;
- ``names_of`` collects every name a member held, and ``until`` leaves out
  intervals that had not begun yet.
"""

from __future__ import annotations

from everos.memory.groups import NamedMember, named_members, names_of
from everos.memory.models import Roster, RosterEvent, fold_roster


def _members(**names: tuple[str, ...]) -> list[NamedMember]:
    return [NamedMember(member_id=m, names=n) for m, n in names.items()]


def test_whole_words_only_and_case_insensitive() -> None:
    members = _members(u_al=("Al",), u_alice=("Alice",))
    assert named_members("alice met AL at noon", members) == ["u_alice", "u_al"]
    assert named_members("Alice called", members) == ["u_alice"]
    assert named_members("Alfred called", members) == []


def test_unspaced_scripts_match_as_substrings() -> None:
    members = _members(u_chen=("陈",), u_wang=("王磊",))
    assert named_members("王磊说陈明天订票", members) == ["u_wang", "u_chen"]


def test_aliases_count_and_order_follows_the_text() -> None:
    members = _members(u_betty=("Betty", "Bets"), u_chen=("Chen",))
    assert named_members("Chen thanked Bets", members) == ["u_chen", "u_betty"]


def test_names_of_respects_the_time_a_member_had_joined() -> None:
    roster = Roster(
        intervals=fold_roster(
            [
                RosterEvent(op="join", member_id="u_dan", at=0, display_name="Dan"),
                RosterEvent(op="leave", member_id="u_dan", at=10),
                RosterEvent(
                    op="join",
                    member_id="u_dan",
                    at=20,
                    display_name="Daniel",
                    aliases=["Danny"],
                ),
            ]
        )
    )
    assert names_of(roster, "u_dan") == ("Dan", "Daniel", "Danny")
    assert names_of(roster, "u_dan", until=15) == ("Dan",)
