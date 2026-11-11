"""A session-filtered search shows buffered messages only to who may see them.

Before the access policy, ``filters.session_id`` returned every buffered
message of that session to any owner id (docs/group-memory.md, section 2,
``manager.py:334-351``). What these pin (section 9.3, row 3; section 10):

- a one-to-one owner sees a session's buffer when they are a sender of it --
  a buffered message of theirs, or memory of theirs from that session;
- anyone else gets an empty list, for any session id they guess;
- a group session's buffer is never readable through a personal scope, and
  through the group scope only inside the asker's windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from everos.memory.models import Episode
from everos.memory.ports import NoSourceCheck, SessionKey
from everos.service import SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

_T0 = 1_772_323_200_000


def _message(sender: str, at: int, text: str) -> dict[str, Any]:
    return {"sender_id": sender, "role": "user", "timestamp": at, "content": text}


async def _buffer(
    world: Any, session: str, *messages: dict[str, Any], **extra: Any
) -> None:
    await world.memory.add(
        {
            "session_id": session,
            "app_id": APP,
            "project_id": PROJECT,
            "messages": list(messages),
            **extra,
        },
        request_id="r",
        async_mode=False,
        defer_extraction=True,
    )


async def _buffered(world: Any, session: str, **scope: Any) -> list[str]:
    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            query="anything",
            filters={"session_id": session},
            **scope,
        )
    )
    return [m.content for m in response.data.unprocessed_messages]


async def test_a_one_to_one_buffer_is_shown_to_its_senders_only(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _buffer(world, "s_alice", _message("u_alice", _T0, "my pin is 4321"))

        assert await _buffered(world, "s_alice", user_id="u_alice") == [
            "my pin is 4321"
        ]
        assert await _buffered(world, "s_alice", user_id="u_mallory") == []


async def test_an_earlier_participant_still_sees_the_tail(tmp_path: Path) -> None:
    """Nothing buffered is theirs, but they have memory from the session."""
    async with GroupWorld.open(tmp_path) as world:
        await world.stores.episode.append(
            world.personal("u_bob"),
            Episode(
                owner_id="u_bob",
                episode="earlier part",
                summary="earlier",
                timestamp=_T0 - 10,
                session_id="s_pair",
                parent_id="mc_x",
            ),
            only_if=NoSourceCheck(),
        )
        await world.project()
        await _buffer(world, "s_pair", _message("u_carol", _T0, "the reply"))

        assert await _buffered(world, "s_pair", user_id="u_bob") == ["the reply"]
        assert await _buffered(world, "s_pair", user_id="u_dave") == []


async def test_a_group_buffer_never_shows_through_a_personal_scope(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await world.roster("trip", [join("u_chen", 0), join("u_evan", _T0 + 5)])
        await world.stores.groups.bind(
            SessionKey(app_id=APP, project_id=PROJECT, session_id="s_group"),
            "trip",
            at=_T0,
        )
        await _buffer(
            world,
            "s_group",
            _message("u_chen", _T0 + 1, "before Evan"),
            _message("u_chen", _T0 + 9, "after Evan"),
        )

        # Chen is a sender of the session, but it belongs to the group.
        assert await _buffered(world, "s_group", user_id="u_chen") == []
        assert await _buffered(
            world, "s_group", group_id="trip", asker_id="u_chen"
        ) == ["before Evan", "after Evan"]
        assert await _buffered(
            world, "s_group", group_id="trip", asker_id="u_evan"
        ) == ["after Evan"]
