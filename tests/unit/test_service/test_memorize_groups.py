"""``/memory/add`` with ``group_id``: binding, roster, implicit joins, the D18 cut.

Real markdown, ``system.db`` and LanceDB, the bootstrap's memory service and a
scripted boundary model (``tests/helpers/groups.py``). What these pin
(docs/group-memory.md, sections 5.2, 7.1, 10 and 14 task 2):

- an unknown ``group_id`` creates the group; an unknown sender joins at their
  first message with ``source = implicit``;
- the first batch binds the session; a batch without ``group_id`` inherits the
  binding; a batch naming another group is 409 and changes nothing;
- a group batch never runs the one-to-one pipeline: nothing lands under
  ``users/`` and no one-to-one event is published;
- an inline roster change cuts the session's open buffer first, so the
  messages buffered before it form their own segment (D18);
- a roster change -- inline on ``/memory/add`` or through the roster route --
  cuts the open buffers of every session of the group before it applies, so
  no segment spans it (task 4);
- without ``group_id`` a multi-sender batch still runs today's path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.core.errors import RosterConflictError
from tests.helpers.groups import APP, PROJECT, GroupWorld

_T0 = 1_772_323_200_000


def _boundary_model(cut_after: int | None = None) -> FakeLLMClient:
    """Cuts nowhere (or after the given message) and never closes the tail.

    A cut segment is the group's episode: the group pipeline extracts it.
    """

    def answer(messages: Any, *, response_format: type, **_kw: Any) -> str:
        if response_format.__name__ == "_EpisodeReply":
            return json.dumps(
                {"title": "Plans", "summary": "The group talked.", "content": "Talk."}
            )
        if response_format.__name__ != "_BatchReply":
            raise AssertionError(f"unexpected model call {response_format.__name__}")
        return json.dumps(
            {
                "reasoning": "scripted",
                "boundaries": [cut_after] if cut_after else [],
                "tail_complete": False,
            }
        )

    return FakeLLMClient(handler=answer)


def _message(sender: str, at: int, text: str, role: str = "user") -> dict[str, Any]:
    return {
        "sender_id": sender,
        "sender_name": sender.split("_")[-1].title(),
        "role": role,
        "timestamp": at,
        "content": text,
    }


def _batch(session: str, *messages: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "session_id": session,
        "app_id": APP,
        "project_id": PROJECT,
        "messages": list(messages),
        **fields,
    }


async def _add(world: Any, payload: dict[str, Any]) -> Any:
    return await world.memory.add(
        payload, request_id="r", async_mode=False, defer_extraction=False
    )


async def test_a_batch_creates_the_group_binds_the_session_and_joins(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=_boundary_model()) as world:
        await _add(
            world,
            _batch(
                "s1",
                _message("u_chen", _T0 + 1, "Let's plan the offsite"),
                _message("bot_ava", _T0 + 2, "Happy to help", role="assistant"),
                group_id="trip",
            ),
        )
        record = await world.stores.groups.get(world.scope("trip"))
        assert record is not None
        assert record.group.created_at == _T0 + 1
        chen = record.roster.current("u_chen")
        ava = record.roster.current("bot_ava")
        assert chen is not None and chen.source == "implicit"
        assert ava is not None and ava.kind == "assistant"
        assert record.sessions == ("s1",)

        # A later batch without group_id inherits the binding.
        await _add(world, _batch("s1", _message("u_betty", _T0 + 3, "Count me in")))
        record = await world.stores.groups.get(world.scope("trip"))
        assert record is not None and record.roster.is_current_member("u_betty")

        # Nothing of the group reached the one-to-one track.
        assert not (tmp_path / "default_app" / PROJECT / "users").exists()
        assert [type(e).__name__ for e in world.bus.events] == []


async def test_a_session_bound_to_one_group_is_409_for_another(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=_boundary_model()) as world:
        await _add(
            world,
            _batch("s1", _message("u_chen", _T0, "hello"), group_id="trip"),
        )
        with pytest.raises(RosterConflictError):
            await _add(
                world,
                _batch("s1", _message("u_chen", _T0 + 1, "hi"), group_id="other"),
            )
        assert await world.stores.groups.get(world.scope("other")) is None


async def test_an_inline_roster_change_cuts_the_open_buffer_first(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=_boundary_model()) as world:
        await _add(
            world,
            _batch(
                "s1",
                _message("u_chen", _T0 + 1, "Before Evan joins"),
                group_id="trip",
                asker_id="u_chen",
            ),
        )
        await _add(
            world,
            _batch(
                "s1",
                _message("u_evan", _T0 + 10, "Hi all, Evan here"),
                group_id="trip",
                asker_id="u_chen",
                roster=[
                    {
                        "op": "join",
                        "member_id": "u_evan",
                        "display_name": "Evan",
                        "at": _T0 + 10,
                    }
                ],
            ),
        )
        buffered = await world.stores.session_buffer.list(world.session("s1"))
        # The message from before the change became its own segment; only the
        # message after it is still buffered.
        assert [m.sender_id for m in buffered] == ["u_evan"]
        record = await world.stores.groups.get(world.scope("trip"))
        assert record is not None
        evan = record.roster.current("u_evan")
        assert evan is not None and (evan.source, evan.joined_at) == (
            "inline",
            _T0 + 10,
        )


async def test_an_inline_roster_change_cuts_every_session_of_the_group(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=_boundary_model()) as world:
        for session in ("s1", "s2"):
            await _add(
                world,
                _batch(
                    session,
                    _message("u_chen", _T0 + 1, f"opening {session}"),
                    group_id="trip",
                    asker_id="u_chen",
                ),
            )
        await _add(
            world,
            _batch(
                "s1",
                _message("u_evan", _T0 + 10, "Hi all, Evan here"),
                group_id="trip",
                asker_id="u_chen",
                roster=[{"op": "join", "member_id": "u_evan", "at": _T0 + 10}],
            ),
        )
        assert [
            m.sender_id
            for m in await world.stores.session_buffer.list(world.session("s1"))
        ] == ["u_evan"]
        # The other session's buffered message became a segment too, before
        # the change, and was extracted as the group's episode.
        assert await world.stores.session_buffer.list(world.session("s2")) == []
        sessions = {
            e.session_id
            for e in world.bus.events
            if type(e).__name__ == "GroupEpisodeExtracted"
        }
        assert sessions == {"s1", "s2"}


async def test_the_roster_route_cuts_every_session_of_the_group(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=_boundary_model()) as world:
        for session in ("s1", "s2"):
            await _add(
                world,
                _batch(
                    session,
                    _message("u_chen", _T0 + 1, f"opening {session}"),
                    group_id="trip",
                    asker_id="u_chen",
                ),
            )
        await world.memory.change_roster(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            asserted_asker=None,
            events=[world.roster_input(op="join", member_id="u_evan", at=_T0 + 50)],
        )
        for session in ("s1", "s2"):
            assert await world.stores.session_buffer.list(world.session(session)) == []


async def test_a_multi_sender_batch_without_group_runs_todays_path(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=_boundary_model()) as world:
        await _add(
            world,
            _batch(
                "s9",
                _message("u_ann", _T0 + 1, "hi"),
                _message("u_bob", _T0 + 2, "hello"),
            ),
        )
        assert await world.stores.groups.binding(world.session("s9")) is None
        buffered = await world.stores.session_buffer.list(world.session("s9"))
        assert [m.sender_id for m in buffered] == ["u_ann", "u_bob"]
