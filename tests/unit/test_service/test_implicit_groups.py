"""Two people are a group too: the implicit group of a session's people.

docs/group-memory.md, section 7.9. Real markdown, ``system.db`` and LanceDB,
the bootstrap's memory service and a scripted model (``tests/helpers/groups.py``).
What these pin:

- a session without ``group_id`` whose messages come from two or more people
  (role ``user`` senders) becomes the implicit group of those people: its id
  is :func:`~everos.memory.groups.implicit_group_id` of their ids, created on
  first sight at the first message with the people and the assistant as
  members, history visible from creation, no owner; the session is bound;
- the same people share one group across their sessions, whatever the order
  they speak in;
- one person first, a second later: the open buffer goes with the group,
  segments already cut stay the first person's own;
- a person who first speaks in a session already bound to an implicit group
  is refused (409) and nothing of the batch is written; a later assistant
  joins;
- a client never names an implicit group on a write: ``/memory/add``,
  ``PUT /groups`` and the roster route answer 422;
- its segments feed the members' profiles in the group and the group profile,
  never a personal profile; nothing lands under ``users/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.core.errors import InvalidInputError, RosterConflictError
from everos.memory.groups import implicit_group_id
from everos.memory.strategies import STRATEGY_SPECS
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

_T0 = 1_772_323_200_000
_PAIR = implicit_group_id(["u_ann", "u_bob"])


class _Model:
    """Cuts after ``cut_after`` buffered messages (never by default); scripted
    episodes, profile updates that add one item per target, an empty group
    profile."""

    def __init__(self, cut_after: int | None = None) -> None:
        self.cut_after = cut_after
        self.calls: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        self.calls.append(name)
        if name == "_BatchReply":
            return json.dumps(
                {
                    "reasoning": "s",
                    "boundaries": [self.cut_after] if self.cut_after else [],
                    "tail_complete": False,
                }
            )
        if name == "EpisodeReply":
            return json.dumps(
                {"title": "Plans", "summary": "They talked.", "content": "Talk."}
            )
        if name == "_ProfileUpdateReply":
            item = {"evidence": "x", "description": "Plans", "category": "plans"}
            return json.dumps(
                {
                    "operations": [
                        {"action": "add", "type": "explicit_info", "data": item}
                    ]
                }
            )
        if name == "GroupProfileReply":
            return json.dumps({"operations": []})
        raise AssertionError(f"unexpected model call {name}")


def _message(sender: str, at: int, text: str, role: str = "user") -> dict[str, Any]:
    return {
        "sender_id": sender,
        "sender_name": sender.split("_")[-1].title(),
        "role": role,
        "timestamp": _T0 + at,
        "content": text,
    }


async def _add(
    world: GroupWorld, session: str, *messages: dict[str, Any], **fields: Any
) -> Any:
    return await world.memory.add(
        {
            "session_id": session,
            "app_id": APP,
            "project_id": PROJECT,
            "messages": list(messages),
            **fields,
        },
        request_id="r",
        async_mode=False,
        defer_extraction=False,
    )


async def _flush(world: GroupWorld, session: str) -> None:
    await world.memory.flush(
        session_id=session, request_id="f", app_id=APP, project_id=PROJECT
    )


def _users_dir(tmp_path: Path) -> Path:
    return tmp_path / "default_app" / PROJECT / "users"


async def test_two_people_without_a_group_become_their_implicit_group(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        await _add(
            world,
            "s1",
            _message("u_ann", 10, "Shall we book the cabin?"),
            _message("bot_ava", 20, "I found two cabins.", role="assistant"),
            _message("u_bob", 30, "The lakeside one."),
        )

        assert await world.stores.groups.binding(world.session("s1")) == _PAIR
        record = await world.stores.groups.get(world.scope(_PAIR))
        assert record is not None
        assert record.group.created_at == _T0 + 10
        assert record.group.created_by is None
        assert record.group.history_visible_from == "creation"
        assert not record.roster.has_managers()
        members = {i.member_id: i for i in record.roster.current_members()}
        assert set(members) == {"u_ann", "u_bob", "bot_ava"}
        assert {i.joined_at for i in members.values()} == {_T0 + 10}
        assert {i.source for i in members.values()} == {"implicit"}
        assert (members["u_ann"].kind, members["u_ann"].role) == ("human", "member")
        assert (members["u_ann"].display_name, members["u_bob"].display_name) == (
            "Ann",
            "Bob",
        )
        assert (members["bot_ava"].kind, members["bot_ava"].role) == (
            "assistant",
            "assistant",
        )

        await _flush(world, "s1")
        (episode,) = [
            e for e in world.bus.events if type(e).__name__ == "EpisodeExtracted"
        ]
        assert (episode.owner_type, episode.owner_id) == ("group", _PAIR)
        assert episode.speaker_ids == ["u_ann", "u_bob"]
        assert not _users_dir(tmp_path).exists()


async def test_the_same_people_share_one_group_across_sessions(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        await _add(
            world, "s1", _message("u_ann", 10, "hi"), _message("u_bob", 20, "hey")
        )
        await _add(
            world, "s2", _message("u_bob", 30, "yo"), _message("u_ann", 40, "hi")
        )

        assert await world.stores.groups.binding(world.session("s2")) == _PAIR
        record = await world.stores.groups.get(world.scope(_PAIR))
        assert record is not None
        assert record.sessions == ("s1", "s2")
        assert record.group.created_at == _T0 + 10


async def test_a_second_person_takes_the_open_buffer_into_the_group(
    tmp_path: Path,
) -> None:
    """The cut segment stays the first person's own; the buffer goes along."""
    model = _Model(cut_after=2)
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        # One person and the assistant: one-to-one, cut after two messages.
        await _add(
            world,
            "s1",
            _message("u_ann", 10, "I want a cabin trip."),
            _message("bot_ava", 20, "Where to?", role="assistant"),
            _message("u_ann", 30, "Somewhere with a lake."),
        )
        assert await world.stores.groups.binding(world.session("s1")) is None
        own = [e for e in world.bus.events if type(e).__name__ == "EpisodeExtracted"]
        assert [(e.owner_type, e.owner_id) for e in own] == [("user", "u_ann")]

        # A second person: the session is now the pair's, from its open buffer.
        model.cut_after = None
        await _add(world, "s1", _message("u_bob", 40, "Count me in."))
        assert await world.stores.groups.binding(world.session("s1")) == _PAIR
        record = await world.stores.groups.get(world.scope(_PAIR))
        assert record is not None
        assert record.group.created_at == _T0 + 30
        assert {i.member_id for i in record.roster.current_members()} == {
            "u_ann",
            "u_bob",
        }
        await _flush(world, "s1")
        group = [
            e
            for e in world.bus.events
            if type(e).__name__ == "EpisodeExtracted" and e.owner_type == "group"
        ]
        assert [e.owner_id for e in group] == [_PAIR]
        cells = await world.stores.memcell.cells_by_id([group[0].memcell_id])
        said = [m.content for m in cells[group[0].memcell_id].items]
        assert said == ["Somewhere with a lake.", "Count me in."]


async def test_a_third_person_in_a_pair_s_session_is_refused(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        await _add(
            world, "s1", _message("u_ann", 10, "hi"), _message("u_bob", 20, "hey")
        )
        before = await world.stores.session_buffer.list(world.session("s1"))

        with pytest.raises(RosterConflictError, match="u_cat"):
            await _add(
                world,
                "s1",
                _message("u_ann", 30, "this is Cat"),
                _message("u_cat", 40, "hello both"),
            )

        assert await world.stores.session_buffer.list(world.session("s1")) == before
        record = await world.stores.groups.get(world.scope(_PAIR))
        assert record is not None
        assert {i.member_id for i in record.roster.intervals} == {"u_ann", "u_bob"}
        # The three of them in a session of their own are a group of their own.
        await _add(
            world,
            "s2",
            _message("u_ann", 50, "hi both"),
            _message("u_cat", 60, "hello"),
            _message("u_bob", 70, "hey"),
        )
        trio = implicit_group_id(["u_cat", "u_bob", "u_ann"])
        assert await world.stores.groups.binding(world.session("s2")) == trio
        assert trio != _PAIR


async def test_a_later_assistant_joins_the_implicit_group(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        await _add(
            world, "s1", _message("u_ann", 10, "hi"), _message("u_bob", 20, "hey")
        )
        await _add(world, "s1", _message("bot_kai", 30, "Hello!", role="assistant"))

        record = await world.stores.groups.get(world.scope(_PAIR))
        assert record is not None
        kai = record.roster.current("bot_kai")
        assert kai is not None
        assert (kai.kind, kai.joined_at) == ("assistant", _T0 + 30)


async def test_a_client_never_names_an_implicit_group_on_a_write(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        with pytest.raises(InvalidInputError, match="implicit group"):
            await _add(world, "s1", _message("u_ann", 10, "hi"), group_id=_PAIR)
        assert await world.stores.groups.get(world.scope(_PAIR)) is None

        await _add(
            world, "s2", _message("u_ann", 10, "hi"), _message("u_bob", 20, "hey")
        )
        with pytest.raises(InvalidInputError, match="implicit group"):
            await world.memory.upsert_group(
                app_id=APP,
                project_id=PROJECT,
                group_id=_PAIR,
                asker_id="u_ann",
                asserted_asker=None,
                name="Cabin trip",
                description=None,
                created_at=None,
                history_visible_from=None,
                extra=None,
                roster=[],
            )
        before = await world.stores.session_buffer.list(world.session("s2"))
        with pytest.raises(InvalidInputError, match="implicit group"):
            await world.memory.change_roster(
                app_id=APP,
                project_id=PROJECT,
                group_id=_PAIR,
                asker_id="u_ann",
                asserted_asker=None,
                events=[world.roster_input(op="join", member_id="u_cat", at=_T0)],
            )
        # Refused before anything happened: the open buffer was not cut.
        assert await world.stores.session_buffer.list(world.session("s2")) == before
        record = await world.stores.groups.get(world.scope(_PAIR))
        assert record is not None and record.group.name == ""
        assert record.roster.current("u_cat") is None


async def test_one_person_stays_one_to_one(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        await _add(
            world,
            "s1",
            _message("u_ann", 10, "hi"),
            _message("bot_ava", 20, "hello", role="assistant"),
        )
        await _flush(world, "s1")

        assert await world.stores.groups.binding(world.session("s1")) is None
        assert await world.stores.groups.group_ids(APP, PROJECT) == []
        (episode,) = [
            e for e in world.bus.events if type(e).__name__ == "EpisodeExtracted"
        ]
        assert (episode.owner_type, episode.owner_id) == ("user", "u_ann")


async def test_its_segments_feed_the_group_s_profiles_never_a_personal_one(
    tmp_path: Path,
) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await world.roster("other", [join("u_ann", _T0)])  # unrelated, untouched
        await _add(
            world,
            "s1",
            _message("u_ann", 10, "I only eat vegetarian."),
            _message("u_bob", 20, "I'll book the lakeside cabin."),
        )
        await _flush(world, "s1")
        await world.run(*STRATEGY_SPECS)

        kinds = [type(e).__name__ for e in world.bus.events]
        assert kinds == ["EpisodeExtracted", "GroupFlushed"]
        for member in ("u_ann", "u_bob"):
            profile = await world.stores.profile.read(world.scope(_PAIR), member)
            assert profile is not None and profile.explicit_info
        assert "GroupProfileReply" in model.calls  # the flush read the batch
        assert not _users_dir(tmp_path).exists()
