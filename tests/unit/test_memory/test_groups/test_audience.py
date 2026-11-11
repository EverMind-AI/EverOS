"""Who knew a group record: ``record_audience`` on roster fixtures (task 16).

docs/group-memory.md, section 8.6. ``present`` holds the members, current or
former, one of whose stays covers the record's whole span; ``joined_after``
the current members who were not there and whose current stay began after
it. Adjacent stays merge, as in the read windows; a rejoin after the record
is not "present".
"""

from __future__ import annotations

from everos.memory.groups import record_audience
from everos.memory.models import Roster, RosterEvent, fold_roster

_ROSTER = Roster(
    intervals=fold_roster(
        [
            RosterEvent(op="join", member_id="u_chen", at=0),
            RosterEvent(op="join", member_id="bot_ava", at=0, kind="assistant"),
            RosterEvent(op="join", member_id="u_betty", at=0),
            RosterEvent(op="leave", member_id="u_betty", at=6_000),
            RosterEvent(op="join", member_id="u_dan", at=0),
            RosterEvent(op="leave", member_id="u_dan", at=2_500),
            RosterEvent(op="join", member_id="u_dan", at=7_500),
            RosterEvent(op="join", member_id="u_evan", at=5_000),
            # Back-to-back stays read as one.
            RosterEvent(op="join", member_id="u_fay", at=0),
            RosterEvent(op="leave", member_id="u_fay", at=3_000),
            RosterEvent(op="join", member_id="u_fay", at=3_000),
        ]
    )
)


def test_who_was_there_and_who_joined_since() -> None:
    early = record_audience(_ROSTER, start=1_000, end=2_000)
    assert early.present == ("bot_ava", "u_betty", "u_chen", "u_dan", "u_fay")
    assert early.joined_after == ("u_evan",)

    across_a_leave = record_audience(_ROSTER, start=2_000, end=3_500)
    assert across_a_leave.present == ("bot_ava", "u_betty", "u_chen", "u_fay")
    # Dan was away for part of it and is back now: he has not heard it.
    assert across_a_leave.joined_after == ("u_dan", "u_evan")

    late = record_audience(_ROSTER, start=8_000, end=9_000)
    assert late.present == ("bot_ava", "u_chen", "u_dan", "u_evan", "u_fay")
    assert late.joined_after == ()  # Betty left; she is no current member


def test_a_record_before_anyone_joined_has_no_one_present() -> None:
    nobody = record_audience(_ROSTER, start=-5, end=-1)
    assert nobody.present == ()
    assert set(nobody.joined_after) == {"bot_ava", "u_chen", "u_dan", "u_evan", "u_fay"}
