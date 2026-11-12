"""Group search and get, the details clients read (docs/group-memory.md, 5.3-5.4).

Real markdown, ``system.db`` and LanceDB (``tests/helpers/groups.py``). What
these pin (task 7):

- a group episode carries ``owner_type = group`` and ``source`` -- its
  memcell and the epoch ms of the segment's first and last message -- and a
  one-to-one episode carries neither;
- ``include_profile`` on a group scope attaches the asker's member-in-group
  profile and those of members the query names (by roster name or alias, at
  most three), never a personal profile, and without the items whose sources
  lie outside the asker's windows;
- the multi-round decider's question is ``"<display name>: <question>"``;
- get lists a member's profile in the group (``member_profile`` with
  ``member_id``) and the group's picture memories (``image_memory``);
- on the wire, the fields read the way the evaluation harnesses read them:
  ``episodes[].source.from`` / ``to`` and ``profiles[].member_id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.app import create_app
from everos.memory.models import ExplicitInfoItem, Profile
from everos.service import GetRequest, SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000


class _Decider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **_: Any) -> str:
        self.prompts.extend(str(m.content) for m in messages)
        return json.dumps({"core": [0], "next_queries": []})


def _item(text: str, start: int, end: int) -> ExplicitInfoItem:
    return ExplicitInfoItem(
        description=text,
        category="plans",
        item_id=f"pi_{start}",
        source_memcell_ids=[f"mc_{start}"],
        source_start=T0 + start,
        source_end=T0 + end,
    )


async def _world(tmp_path: Path, decider: Any = None) -> Any:
    cm = GroupWorld.open(tmp_path, decider=decider)
    world = await cm.__aenter__()
    await world.roster(
        "trip",
        [
            join("u_chen", T0, display_name="Chen", role="owner"),
            join("u_betty", T0, display_name="Betty", aliases=["Bets"]),
            join("u_evan", T0 + 5_000, display_name="Evan"),
        ],
    )
    await world.episode(
        "trip", "Chen booked the Lisbon venue", start=T0 + 1_000, end=T0 + 2_000
    )
    await world.personal_episode("u_chen", "Chen likes Lisbon tiles", at=T0 + 1_000)
    await world.profile("u_chen", "Chen lives in Porto")
    for member, items in (
        ("u_chen", [_item("Organises the Lisbon trip", 1_000, 2_000)]),
        (
            "u_betty",
            [
                _item("Books the Lisbon hotel", 1_000, 2_000),
                _item("Prefers aisle seats", 6_000, 7_000),
            ],
        ),
    ):
        await world.stores.profile.write(
            world.scope("trip"),
            Profile(
                owner_id=member,
                summary=f"{member} in the trip group",
                timestamp=T0 + 7_000,
                explicit_info=items,
            ),
        )
    await world.project()
    return world, cm


def _search(asker: str, query: str, **fields: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=asker,
        query=query,
        **fields,
    )


async def test_a_group_episode_carries_its_owner_type_and_span(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        found = await world.memory.search(_search("u_chen", "Lisbon venue", top_k=5))
        (episode,) = found.data.episodes
        assert episode.owner_type == "group"
        assert episode.group_id == "trip"
        assert episode.source is not None
        data = episode.model_dump(mode="json", by_alias=True)["source"]
        assert data == {
            "memcell_id": f"mc_trip_{T0 + 1_000}",
            "from": T0 + 1_000,
            "to": T0 + 2_000,
        }
        personal = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="u_chen",
                query="Lisbon tiles",
            )
        )
        (mine,) = personal.data.episodes
        assert mine.owner_type is None and mine.source is None
    finally:
        await cm.__aexit__(None, None, None)


async def test_include_profile_attaches_the_askers_and_named_members(
    tmp_path: Path,
) -> None:
    world, cm = await _world(tmp_path)
    try:
        found = await world.memory.search(
            _search(
                "u_chen",
                "What does Bets still have to book?",
                include_profile=True,
            )
        )
        profiles = {p.member_id: p for p in found.data.profiles}
        assert set(profiles) == {"u_chen", "u_betty"}
        for profile in found.data.profiles:
            assert (profile.owner_type, profile.group_id, profile.user_id) == (
                "group",
                "trip",
                None,
            )
        blob = json.dumps(found.data.model_dump(mode="json"))
        assert "Lives in Porto" not in blob and "Porto" not in blob

        # Evan joined at 5000: Betty's early item is outside his window, and
        # a profile with an invisible item shows no summary.
        evan = await world.memory.search(
            _search("u_evan", "Betty seats", include_profile=True)
        )
        (betty,) = [p for p in evan.data.profiles if p.member_id == "u_betty"]
        items = [i["description"] for i in betty.profile_data["explicit_info"]]
        assert items == ["Prefers aisle seats"]
        assert betty.profile_data["summary"] == ""
        assert "u_chen" not in {p.member_id for p in evan.data.profiles}
    finally:
        await cm.__aexit__(None, None, None)


async def test_the_deciders_question_names_the_asker(tmp_path: Path) -> None:
    decider = _Decider()
    world, cm = await _world(tmp_path, decider=FakeLLMClient(handler=decider))
    try:
        await world.memory.search(_search("u_betty", "Who booked the venue?"))
        assert any("Betty: Who booked the venue?" in p for p in decider.prompts)
    finally:
        await cm.__aexit__(None, None, None)


async def test_get_lists_member_profiles_and_group_pictures(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        await world.picture(
            world.scope("trip"), "Floor plan of the venue", at=T0 + 1_500, shade=7
        )
        await world.project()
        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="member_profile",
                member_id="u_betty",
            )
        )
        (profile,) = listed.data.profiles
        assert (profile.member_id, profile.group_id) == ("u_betty", "trip")
        pictures = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="image_memory",
            )
        )
        (picture,) = pictures.data.images
        assert picture.description == "Floor plan of the venue"
        assert picture.sender_ids == ["u_chen"]
        # Evan joined later: the picture is outside his window.
        evan = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_evan",
                memory_type="image_memory",
            )
        )
        assert evan.data.images == []
    finally:
        await cm.__aexit__(None, None, None)


def test_member_profile_needs_a_member_and_a_group_scope() -> None:
    with pytest.raises(ValueError):
        GetRequest(group_id="trip", asker_id="u_chen", memory_type="member_profile")
    with pytest.raises(ValueError):
        GetRequest(user_id="u_chen", memory_type="member_profile", member_id="u_b")


async def test_the_wire_shape_the_harnesses_read(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        app = create_app(application=make_application(memory=world.memory))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            response = await client.post(
                "/api/v2/memory/search",
                json={
                    "app_id": APP,
                    "project_id": PROJECT,
                    "group_id": "trip",
                    "asker_id": "u_chen",
                    "query": "Where does Bets stay in Lisbon?",
                    "method": "llm_multiround",
                    "include_profile": True,
                },
            )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        (episode,) = data["episodes"]
        assert episode["owner_type"] == "group"
        assert episode["source"] == {
            "memcell_id": f"mc_trip_{T0 + 1_000}",
            "from": T0 + 1_000,
            "to": T0 + 2_000,
        }
        for key in ("sender_ids", "subject_ids", "origin", "subject", "timestamp"):
            assert key in episode
        members = {p["member_id"] for p in data["profiles"]}
        assert members == {"u_chen", "u_betty"}
        assert {p["owner_type"] for p in data["profiles"]} == {"group"}
    finally:
        await cm.__aexit__(None, None, None)


async def test_named_members_are_counted_once_across_a_pool(tmp_path: Path) -> None:
    """A member of several groups of the pool is one of the three named."""
    world, cm = await _world(tmp_path)
    try:
        await world.roster("trip", [join("u_dan", T0, display_name="Dan")])
        await world.roster(
            "ops",
            [
                join("u_chen", T0, display_name="Chen"),
                join("u_betty", T0, display_name="Betty"),
            ],
        )
        for group, member in (
            ("ops", "u_betty"),
            ("trip", "u_dan"),
            ("trip", "u_evan"),
        ):
            await world.stores.profile.write(
                world.scope(group),
                Profile(
                    owner_id=member,
                    summary="",
                    timestamp=T0 + 9_000,
                    explicit_info=[_item(f"{member} in {group}", 6_000, 7_000)],
                ),
            )
        await world.project()
        found = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                group_ids=["trip", "ops"],
                asker_id="u_chen",
                query="What do Betty, Dan and Evan still owe?",
                include_profile=True,
            )
        )
        attached = sorted((p.group_id, p.member_id) for p in found.data.profiles)
        assert attached == [
            ("ops", "u_betty"),
            ("trip", "u_betty"),
            ("trip", "u_chen"),
            ("trip", "u_dan"),
            ("trip", "u_evan"),
        ]
    finally:
        await cm.__aexit__(None, None, None)
