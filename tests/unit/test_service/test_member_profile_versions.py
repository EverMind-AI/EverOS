"""A member's fact that changes value, end to end (docs/group-memory.md, 7.4).

A group chat goes through the bootstrap's memory service (add, flush), the
``update_member_profiles`` strategy runs everalgo's real merge on a scripted
model, and the cascade projects the profile into the index. Betty first says
she lives in Paris, later that she moved to Berlin; the model's update
modifies the Paris item by its ref. What these pin:

- the stored Berlin item names the Paris one and when it changed, and the
  Paris history entry names Berlin back;
- a search and a get show Berlin with ``since`` and ``previous: Paris (until
  ...)`` to a member who saw both;
- a member who joined between the two sees Berlin as a plain item: no
  previous value, no ``since``, no ``supersedes`` naming what they cannot see;
- a forgotten value is never shown again, as "previously" either: not after
  its message is forgotten, not after it is forgotten by its own id; undo
  shows it again.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.memory.strategies import UPDATE_MEMBER_PROFILES_SPEC
from everos.service import GetRequest, GroupForgetTarget, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000
_TARGET = re.compile(r"The target user is (.+?), the speaker")
_NOTHING_ON_FILE = "<explicit_info>\n(nothing on file)\n</explicit_info>"


def _fact(description: str) -> dict[str, str]:
    return {"evidence": "said so", "description": description, "category": "home"}


def _model(messages: Any, *, response_format: type, **_: Any) -> str:
    name = response_format.__name__
    prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
    if name == "_BatchReply":
        return json.dumps({"reasoning": "s", "boundaries": [], "tail_complete": False})
    if name == "EpisodeReply":
        return json.dumps({"title": "Chat", "summary": "A chat.", "content": "Chat."})
    found = _TARGET.search(prompt)
    who = found.group(1).strip() if found else "?"
    if name == "_ProfileUpdateReply" and _NOTHING_ON_FILE in prompt:
        fact = "Lives in Paris" if who == "Betty" else "Is friendly"
        add = {"action": "add", "type": "explicit_info", "data": _fact(fact)}
        return json.dumps({"operations": [add]})
    if name == "_ProfileUpdateReply":
        ops: list[dict[str, Any]] = [{"action": "none"}]
        if who == "Betty" and "Berlin" in prompt:
            ops = [{"action": "modify", "ref": "e1", "data": _fact("Lives in Berlin")}]
        return json.dumps({"operations": ops})
    raise AssertionError(f"unexpected model call {name}")


async def _say(world: GroupWorld, mid: str, who: str, text: str, at: int) -> None:
    await world.memory.add(
        {
            "session_id": "s_trip",
            "app_id": APP,
            "project_id": PROJECT,
            "group_id": "trip",
            "messages": [
                {
                    "message_id": mid,
                    "sender_id": who,
                    "role": "user",
                    "timestamp": T0 + at,
                    "content": text,
                }
            ],
        },
        request_id="r",
        async_mode=False,
        defer_extraction=True,
    )
    await _flush(world)


async def _flush(world: GroupWorld) -> None:
    await world.memory.flush(
        session_id="s_trip", request_id="f", app_id=APP, project_id=PROJECT
    )
    await world.run(UPDATE_MEMBER_PROFILES_SPEC)
    await world.project()


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[GroupWorld]:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_model)) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, role="owner", display_name="Chen"),
                join("u_betty", T0, display_name="Betty"),
                join("u_evan", T0 + 3_000, display_name="Evan"),
            ],
        )
        await _say(world, "m-1", "u_betty", "I live in Paris", 1_000)
        await _say(world, "m-2", "u_chen", "Hello all", 2_000)
        await _say(world, "m-3", "u_betty", "We moved to Berlin last week", 5_000)
        yield world


async def _betty(world: GroupWorld, asker: str) -> dict[str, Any]:
    """Betty's home item as a get by ``asker`` returns it."""
    response = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            memory_type="member_profile",
            member_id="u_betty",
        )
    )
    (profile,) = response.model_dump(mode="json")["data"]["profiles"]
    (item,) = profile["profile_data"]["explicit_info"]
    return item


async def _search(world: GroupWorld, asker: str, part: str | None = None) -> str:
    """The search response by ``asker``, or only its ``part`` (``profiles``)."""
    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            query="Where does Betty live?",
            top_k=10,
            group_id="trip",
            asker_id=asker,
            include_profile=True,
        )
    )
    data = response.data.model_dump(mode="json")
    return json.dumps(data if part is None else data[part])


async def test_the_new_value_names_the_old_one_and_reads_show_both(
    world: GroupWorld,
) -> None:
    held = await world.stores.member_profile.read(world.scope("trip"), "u_betty")
    assert held is not None
    (berlin,) = held.profile.explicit_info
    (paris,) = held.history
    extra = berlin.model_extra or {}
    assert berlin.description == "Lives in Berlin"
    assert extra["supersedes"] == paris["item_id"]
    assert extra["since"] == paris["replaced_at"] == extra["source_end"]
    assert paris["superseded_by"] == extra["item_id"]

    item = await _betty(world, "u_chen")
    assert item["since"] == extra["since"]
    assert item["previous"] == [
        {
            "item_id": paris["item_id"],
            "description": "Lives in Paris",
            "until": extra["since"],
            "source_start": paris["source_start"],
            "source_end": paris["source_end"],
        }
    ]
    found = await _search(world, "u_chen")
    assert '"previous"' in found and "Lives in Paris" in found


async def test_a_member_who_joined_later_sees_no_version(world: GroupWorld) -> None:
    item = await _betty(world, "u_evan")
    assert item["description"] == "Lives in Berlin"
    assert not {"previous", "since", "supersedes"} & set(item)
    assert "Paris" not in await _search(world, "u_evan")


async def test_a_forgotten_value_is_never_shown_as_previous(
    world: GroupWorld,
) -> None:
    async def forget(target: GroupForgetTarget) -> Any:
        result = await world.memory.forget_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            target=target,
        )
        await _flush(world)
        return result

    async def undo(forget_id: str) -> None:
        await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            forget_id=forget_id,
        )
        await world.project()

    # The message that stated the old value.
    message = GroupForgetTarget(
        kind="message", session_id="s_trip", message_ids=("m-1",)
    )
    result = await forget(message)
    item = await _betty(world, "u_chen")
    assert item["description"] == "Lives in Berlin"
    assert not {"previous", "since", "supersedes"} & set(item)
    assert "Paris" not in await _search(world, "u_chen")
    await undo(result.forget_id)
    assert (await _betty(world, "u_chen"))["previous"][0]["description"] == (
        "Lives in Paris"
    )

    # The old value by its own id, as a read lists it. The message stays.
    old_id = (await _betty(world, "u_chen"))["previous"][0]["item_id"]
    result = await forget(GroupForgetTarget(kind="record", record_ids=(old_id,)))
    assert result.member_profile_items == (old_id,)
    assert "previous" not in await _betty(world, "u_chen")
    assert "Paris" not in await _search(world, "u_chen", "profiles")
    assert "I live in Paris" in await _search(world, "u_chen")
    held = await world.stores.member_profile.read(world.scope("trip"), "u_betty")
    assert held is not None and held.history == []
