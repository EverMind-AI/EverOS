"""Who may forget what inside a group, and what a read skips once forgotten.

docs/group-memory.md, section 5.5 (the table of targets) and section 9.4.
What these pin:

- a former member and a non-member are refused every forget (403);
- a member may forget only what rests on their own messages (their messages,
  themselves, records whose sources are all theirs, pictures they posted); a
  target resting on anyone else's, or on nobody's (a whole session), is the
  owners' and admins';
- owners and admins may forget anything, and in a group without either any
  current member may, as for roster changes;
- a forgotten message is absent from a session's buffer read, and from a
  segment read through :func:`without_forgotten`; a forgotten picture leaves
  the messages that carried it;
- a picture nothing refers to any more is served by hash unless a group
  forgot it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from everalgo.types import ChatMessage, MemCell, ToolCallResult

from everos.core.errors import ForbiddenError
from everos.memory.access import (
    AccessPolicy,
    Identity,
    ScopeRequest,
    authorize_forget,
)
from everos.memory.extract.pipeline.pictures import without_pictures
from everos.memory.groups import (
    readable_segments,
    without_forgotten,
)
from everos.memory.models import (
    CanonicalMessage,
    Group,
    Roster,
    RosterEvent,
    fold_roster,
)
from everos.memory.ports import Scope, SessionKey
from tests.fakes import InMemoryGroupStore
from tests.fakes.stores import InMemoryMessageForgetStore

_SESSION = SessionKey(app_id="default", project_id="p", session_id="s1")


def _roster(*events: RosterEvent) -> Roster:
    return Roster(intervals=fold_roster(events))


_GROUP = _roster(
    RosterEvent(op="join", member_id="u_owner", at=0, role="owner"),
    RosterEvent(op="join", member_id="u_admin", at=0, role="admin"),
    RosterEvent(op="join", member_id="u_betty", at=0),
    RosterEvent(op="join", member_id="u_dan", at=0),
    RosterEvent(op="join", member_id="u_left", at=0),
    RosterEvent(op="leave", member_id="u_left", at=10),
)


@pytest.mark.parametrize("asker", ["u_left", "u_zoe", None])
def test_a_former_or_non_member_may_forget_nothing(asker: str | None) -> None:
    for sources in ((), (asker or "x",), ("u_betty",)):
        with pytest.raises(ForbiddenError):
            authorize_forget(_GROUP, asker, sources=sources)


def test_a_member_forgets_only_what_rests_on_their_own_messages() -> None:
    authorize_forget(_GROUP, "u_betty", sources=["u_betty"])
    authorize_forget(_GROUP, "u_betty", sources=["u_betty", "u_betty"])
    for sources in (["u_dan"], ["u_betty", "u_dan"], []):
        with pytest.raises(ForbiddenError):
            authorize_forget(_GROUP, "u_betty", sources=sources)


@pytest.mark.parametrize("manager", ["u_owner", "u_admin"])
def test_owners_and_admins_forget_anything(manager: str) -> None:
    for sources in ([], ["u_betty"], ["u_left", "u_dan"], ["someone_gone"]):
        authorize_forget(_GROUP, manager, sources=sources)


def test_a_group_without_managers_is_managed_by_any_member() -> None:
    flat = _roster(
        RosterEvent(op="join", member_id="u_a", at=0),
        RosterEvent(op="join", member_id="u_b", at=0),
    )
    authorize_forget(flat, "u_a", sources=["u_b"])
    authorize_forget(flat, "u_a", sources=[])


def _message(message_id: str, at: int, client_id: str | None) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=message_id,
        session_id="s1",
        sender_id="u_betty",
        role="user",
        timestamp=dt.datetime.fromtimestamp(at, tz=dt.UTC),
        text=f"text of {message_id}",
        client_message_id=client_id,
    )


async def test_a_forgotten_message_is_absent_from_the_buffer() -> None:
    groups = InMemoryGroupStore()
    scope = Scope(app_id="default", project_id="p", owner_id="g", owner_type="group")
    await groups.save(
        scope,
        group=Group(group_id="g", created_at=0),
        events=[RosterEvent(op="join", member_id="u_betty", at=0)],
    )
    await groups.bind(_SESSION, "g", at=0)
    forgotten = InMemoryMessageForgetStore()
    await forgotten.record(_SESSION, ["m-2", "gen-3"], forget_id="fg_1")
    policy = AccessPolicy(groups, forgotten=forgotten)
    read = await policy.read_scope(
        Identity(asker_id="u_betty"),
        ScopeRequest(project_id="p", group_ids=("g",)),
    )
    buffered = [
        _message("gen-1", 100, "m-1"),
        _message("gen-2", 101, "m-2"),  # forgotten by its client id
        _message("gen-3", 102, None),  # forgotten by its generated id
    ]

    async def unused() -> bool:
        raise AssertionError("a group buffer read never asks this")

    shown = await policy.buffered_messages(
        read, _SESSION, buffered, owner_has_memory=unused
    )
    assert [m.message_id for m in shown] == ["gen-1"]
    await forgotten.undo("fg_1")
    shown = await policy.buffered_messages(
        read, _SESSION, buffered, owner_has_memory=unused
    )
    assert len(shown) == 3


def _cell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m-1", role="user", sender_id="a", content="one", timestamp=1
            ),
            ChatMessage(
                id="m-2",
                role="user",
                sender_id="b",
                content=f"look [IMAGE: {'aa' * 32}]",
                timestamp=2,
            ),
            ChatMessage(
                id="m-3",
                role="user",
                sender_id="a",
                content=f"[IMAGE: {'aa' * 32}]",
                timestamp=3,
            ),
            ToolCallResult(tool_call_id="t", content="tool output", timestamp=4),
        ],
        timestamp=4,
    )


def test_a_segment_is_read_without_its_forgotten_messages() -> None:
    cell = _cell()
    read = without_forgotten(cell, {"m-1", "m-9"})
    assert [getattr(i, "id", None) for i in read.items] == ["m-2", "m-3", None]
    assert read.timestamp == cell.timestamp
    assert without_forgotten(cell, ()) is cell
    assert len(cell.items) == 4  # the payload itself is untouched


def test_a_forgotten_picture_leaves_the_messages_that_carried_it() -> None:
    read = without_pictures(_cell(), ["aa" * 32])
    texts = [getattr(i, "content", None) for i in read.items]
    assert texts == ["one", "look", "tool output"]  # m-3 held only the picture
    assert without_pictures(_cell(), ["bb" * 32]).items == _cell().items


async def test_readable_segments_reads_each_without_its_forgotten() -> None:
    forgotten = InMemoryMessageForgetStore()
    forgotten.segments = {"mc_1": _SESSION}
    await forgotten.record(_SESSION, ["m-1"], forget_id="fg_1")
    read = await readable_segments(forgotten, {"mc_1": _cell()})
    assert [getattr(i, "id", None) for i in read["mc_1"].items] == ["m-2", "m-3", None]


async def test_a_hash_a_group_forgot_is_served_to_nobody() -> None:
    policy = AccessPolicy(InMemoryGroupStore())
    asker = Identity(asker_id="u_betty")
    assert await policy.picture_visible(asker, [])
    assert not await policy.picture_visible(asker, [], forgotten_in_group=True)
    assert not await policy.picture_visible(Identity(), [], forgotten_in_group=True)
