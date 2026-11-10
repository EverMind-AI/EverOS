"""RosterManager: who may change a group, and what a batch adds to its roster.

What these pin (docs/group-memory.md, sections 5.1-5.2 and 14 task 2):

- creating a group makes the asker its owner unless the roster names one;
- meta and roster are managed by owners and admins; a member may leave by
  themselves, but not backdated; a non-member is 403 and so is a missing group;
- an unknown ``group_id`` on a batch creates the group at its first message,
  and an unknown sender joins at their first message with ``source =
  implicit`` (an assistant sender as an assistant member);
- a session bound to one group answers 409 for another; with
  ``implicit_join`` off an unknown sender is 422;
- ``data/team_chat_en.json``'s ``user_details`` imports as a roster of five
  members with their titles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.core.errors import ForbiddenError, InvalidInputError, RosterConflictError
from everos.memory.groups import BatchMessage, GroupMetaUpdate, RosterManager
from everos.memory.models import RosterEvent
from everos.memory.ports import Scope, SessionKey
from tests.fakes import InMemoryGroupStore

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_SESSION = SessionKey(app_id="default", project_id="p", session_id="trip-main")
_T0 = 1_772_323_200_000


def _manager(**flags: bool) -> RosterManager:
    return RosterManager(InMemoryGroupStore(), clock=lambda: _T0 + 999, **flags)


async def test_creating_a_group_makes_the_asker_its_owner() -> None:
    rosters = _manager()
    change = await rosters.upsert(
        _SCOPE,
        asker_id="u_chen",
        meta=GroupMetaUpdate(name="Trip team", created_at=_T0),
        roster=[RosterEvent(op="join", member_id="u_betty", at=_T0)],
    )

    assert change.created
    chen = change.record.roster.current("u_chen")
    assert chen is not None and chen.role == "owner" and chen.joined_at == _T0
    assert change.record.group.created_by == "u_chen"
    assert change.record.group.name == "Trip team"


async def test_an_asker_already_in_the_roster_is_promoted_not_duplicated() -> None:
    change = await _manager().upsert(
        _SCOPE,
        asker_id="u_chen",
        meta=GroupMetaUpdate(created_at=_T0),
        roster=[RosterEvent(op="join", member_id="u_chen", at=_T0, title="Organiser")],
    )
    (chen,) = change.record.roster.intervals_of("u_chen")
    assert (chen.role, chen.title) == ("owner", "Organiser")


async def _team(rosters: RosterManager) -> None:
    await rosters.upsert(
        _SCOPE,
        asker_id="u_chen",
        meta=GroupMetaUpdate(created_at=_T0),
        roster=[
            RosterEvent(op="join", member_id="u_ann", role="admin", at=_T0),
            RosterEvent(op="join", member_id="u_betty", at=_T0),
        ],
    )


async def test_meta_and_roster_need_an_owner_or_admin() -> None:
    rosters = _manager()
    await _team(rosters)

    with pytest.raises(ForbiddenError):
        await rosters.upsert(
            _SCOPE, asker_id="u_betty", meta=GroupMetaUpdate(name="Mine now")
        )
    with pytest.raises(ForbiddenError):
        await rosters.change(
            _SCOPE,
            asker_id="u_betty",
            events=[RosterEvent(op="join", member_id="u_eve", at=_T0 + 5)],
        )
    with pytest.raises(ForbiddenError):
        await rosters.change(
            _SCOPE,
            asker_id="u_zoe",
            events=[RosterEvent(op="leave", member_id="u_zoe", at=_T0 + 5)],
        )
    renamed = await rosters.upsert(
        _SCOPE, asker_id="u_ann", meta=GroupMetaUpdate(name="Trip")
    )
    assert renamed.record.group.name == "Trip"
    assert not renamed.created


async def test_a_member_may_leave_but_not_backdated() -> None:
    rosters = _manager()
    await _team(rosters)
    await rosters.change(
        _SCOPE,
        asker_id="u_chen",
        events=[RosterEvent(op="join", member_id="u_evan", at=_T0 + 100)],
    )

    with pytest.raises(ForbiddenError):
        await rosters.change(
            _SCOPE,
            asker_id="u_betty",
            events=[RosterEvent(op="leave", member_id="u_betty", at=_T0 + 50)],
        )
    left = await rosters.change(
        _SCOPE,
        asker_id="u_betty",
        events=[RosterEvent(op="leave", member_id="u_betty", at=_T0 + 200)],
    )
    assert not left.record.roster.is_current_member("u_betty")


async def test_a_group_reads_for_current_members_only() -> None:
    rosters = _manager()
    await _team(rosters)
    await rosters.change(
        _SCOPE,
        asker_id="u_betty",
        events=[RosterEvent(op="leave", member_id="u_betty", at=_T0 + 200)],
    )

    assert (await rosters.read(_SCOPE, asker_id="u_chen")).group.group_id == "trip"
    for asker, scope in (
        ("u_betty", _SCOPE),
        (
            "u_chen",
            Scope(app_id="default", project_id="p", owner_id="x", owner_type="group"),
        ),
    ):
        with pytest.raises(ForbiddenError):
            await rosters.read(scope, asker_id=asker)


def _messages(*rows: tuple[str, str, int]) -> list[BatchMessage]:
    return [
        BatchMessage(sender_id=s, role=r, at=t, sender_name=s.title())
        for s, r, t in rows
    ]


async def test_a_batch_creates_its_group_and_joins_its_senders() -> None:
    rosters = _manager()
    admitted = await rosters.admit_batch(
        _SCOPE,
        session=_SESSION,
        asker_id=None,
        roster=[],
        messages=_messages(
            ("u_chen", "user", _T0 + 10),
            ("bot_ava", "assistant", _T0 + 20),
            ("u_chen", "user", _T0 + 30),
        ),
    )

    assert admitted.created
    assert admitted.record.group.created_at == _T0 + 10
    assert admitted.record.group.created_by is None
    chen = admitted.record.roster.current("u_chen")
    ava = admitted.record.roster.current("bot_ava")
    assert chen is not None and (chen.source, chen.joined_at) == ("implicit", _T0 + 10)
    assert ava is not None and (ava.kind, ava.role) == ("assistant", "assistant")
    assert await rosters.store.binding(_SESSION) == "trip"


async def test_a_session_bound_to_one_group_is_409_for_another() -> None:
    rosters = _manager()
    await rosters.admit_batch(
        _SCOPE,
        session=_SESSION,
        asker_id=None,
        roster=[],
        messages=_messages(("u_chen", "user", _T0)),
    )
    other = Scope(
        app_id="default", project_id="p", owner_id="other", owner_type="group"
    )

    with pytest.raises(RosterConflictError):
        await rosters.admit_batch(
            other,
            session=_SESSION,
            asker_id=None,
            roster=[],
            messages=_messages(("u_chen", "user", _T0 + 1)),
        )
    assert await rosters.store.get(other) is None  # nothing was created


async def test_implicit_join_off_refuses_an_unknown_sender() -> None:
    rosters = _manager(implicit_join=False)
    await rosters.upsert(
        _SCOPE, asker_id="u_chen", meta=GroupMetaUpdate(created_at=_T0)
    )

    with pytest.raises(InvalidInputError):
        await rosters.admit_batch(
            _SCOPE,
            session=_SESSION,
            asker_id=None,
            roster=[],
            messages=_messages(("u_chen", "user", _T0 + 1), ("u_eve", "user", _T0 + 2)),
        )


async def test_inline_roster_for_an_existing_group_needs_a_manager() -> None:
    rosters = _manager()
    await _team(rosters)
    join_evan = [
        RosterEvent(op="join", member_id="u_evan", at=_T0 + 5, source="inline")
    ]

    with pytest.raises(ForbiddenError):
        await rosters.admit_batch(
            _SCOPE, session=_SESSION, asker_id="u_betty", roster=join_evan, messages=[]
        )
    admitted = await rosters.admit_batch(
        _SCOPE, session=_SESSION, asker_id="u_ann", roster=join_evan, messages=[]
    )
    assert admitted.record.roster.is_current_member("u_evan")


async def test_a_late_message_rejoins_a_former_member_an_early_one_does_not() -> None:
    rosters = _manager()
    await _team(rosters)
    await rosters.change(
        _SCOPE,
        asker_id="u_chen",
        events=[RosterEvent(op="leave", member_id="u_betty", at=_T0 + 100)],
    )

    early = await rosters.admit_batch(
        _SCOPE,
        session=_SESSION,
        asker_id=None,
        roster=[],
        messages=_messages(("u_betty", "user", _T0 + 50)),
    )
    assert early.appended == []
    late = await rosters.admit_batch(
        _SCOPE,
        session=_SESSION,
        asker_id=None,
        roster=[],
        messages=_messages(("u_betty", "user", _T0 + 300)),
    )
    assert [(e.op, e.at, e.source) for e in late.appended] == [
        ("join", _T0 + 300, "implicit")
    ]
    assert late.record.roster.windows_of("u_betty") == [
        (_T0, _T0 + 100),
        (_T0 + 300, None),
    ]


def _team_chat() -> dict:
    path = Path(__file__).resolve().parents[4] / "data" / "team_chat_en.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_team_chat_user_details_import_as_a_roster_of_five() -> None:
    data = _team_chat()
    details = data["session_meta"]["user_details"]
    events = [
        RosterEvent(
            op="join",
            member_id=member_id,
            kind="assistant" if detail.get("role") == "assistant" else "human",
            display_name=detail["full_name"],
            title=detail.get("custom_role", ""),
            at=_T0,
        )
        for member_id, detail in details.items()
    ]
    change = await _manager().upsert(
        _SCOPE,
        asker_id="user_103",
        meta=GroupMetaUpdate(name=data["session_meta"]["name"], created_at=_T0),
        roster=events,
    )

    members = change.record.roster.current_members()
    assert len(members) == 5
    by_id = {m.member_id: m for m in members}
    assert by_id["user_103"].title == "Product Owner/Project Initiator"
    assert by_id["user_103"].role == "owner"  # the asker, promoted
    assert by_id["user_102"].display_name == "Betty"
    assert all(m.title for m in members)
