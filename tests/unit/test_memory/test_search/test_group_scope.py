"""Search and get over a group scope, through the access policy.

Real markdown, ``system.db`` and LanceDB (``tests/helpers/groups.py``). What
these pin (docs/group-memory.md, sections 5.3-5.4, 8.1 and 14 task 7):

- two askers with different membership windows get different results for one
  query, on the multi-round route and on get;
- a non-member is 403, and so is a personal read of someone else's scope;
- ``group_ids`` searches several groups as one ranked pool, each with its own
  windows;
- a personal scope never returns a group record, and a group scope never a
  personal one; a group scope attaches no personal profile; a method other
  than ``llm_multiround`` is refused;
- a picture an episode refers to is attached only when the asker may see it;
- the response carries the scope block with the asker's windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from everos.core.errors import ForbiddenError, InvalidInputError
from everos.service import GetRequest, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

_EARLY = "The team picked Lisbon for the offsite venue"
_LATE = "The team moved the Lisbon offsite to April thirtieth"


async def _world(tmp_path: Path) -> Any:
    world_cm = GroupWorld.open(tmp_path)
    world = await world_cm.__aenter__()
    await world.roster(
        "trip",
        [
            join("u_chen", 0, role="owner"),
            join("u_evan", 5_000),  # a late joiner
            join("u_betty", 0),
            leave("u_betty", 6_000),  # left after the late segment began
        ],
    )
    await world.roster("ops", [join("u_chen", 0)])
    await world.episode("trip", _EARLY, start=1_000, end=2_000)
    await world.episode("trip", _LATE, start=7_000, end=8_000)
    await world.episode("ops", "Lisbon ops budget approved", start=1_000, end=1_500)
    await world.personal_episode("u_chen", "Chen likes Lisbon tiles", at=1_000)
    await world.profile("u_chen", "Lives in Porto")
    await world.project()
    return world, world_cm


def _search(**fields: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP, project_id=PROJECT, query="Lisbon offsite", **fields
    )


def _texts(response: Any) -> set[str]:
    return {e.episode for e in response.data.episodes}


async def test_askers_with_different_windows_see_different_results(
    tmp_path: Path,
) -> None:
    world, cm = await _world(tmp_path)
    try:
        chen = await world.memory.search(
            _search(group_id="trip", asker_id="u_chen", top_k=10)
        )
        evan = await world.memory.search(
            _search(group_id="trip", asker_id="u_evan", top_k=10)
        )
    finally:
        await cm.__aexit__(None, None, None)

    assert _texts(chen) <= {_EARLY, _LATE} and _EARLY in _texts(chen)
    assert _texts(evan) <= {_LATE}
    for episode in [*chen.data.episodes, *evan.data.episodes]:
        assert episode.group_id == "trip" and episode.user_id is None
    assert chen.data.scope is not None
    assert [(w.group_id, w.from_, w.to) for w in evan.data.scope.windows] == [
        ("trip", 5_000, None)
    ]
    assert chen.data.profiles == []


async def test_non_members_and_other_owners_are_refused(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        for asker in ("u_betty", "u_zoe"):  # a former member, a stranger
            with pytest.raises(ForbiddenError):
                await world.memory.search(_search(group_id="trip", asker_id=asker))
        with pytest.raises(ForbiddenError):
            await world.memory.search(_search(user_id="u_chen", asker_id="u_evan"))
        with pytest.raises(InvalidInputError):
            await world.memory.search(_search(group_id="trip"))
        with pytest.raises(ValidationError, match="llm_multiround"):
            _search(group_id="trip", asker_id="u_chen", method="agentic")
    finally:
        await cm.__aexit__(None, None, None)


async def test_personal_and_group_memory_never_mix(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        personal = await world.memory.search(
            _search(user_id="u_chen", include_profile=True)
        )
        group = await world.memory.search(
            _search(
                group_id="trip",
                asker_id="u_chen",
                include_profile=True,
            )
        )
    finally:
        await cm.__aexit__(None, None, None)

    assert _texts(personal) == {"Chen likes Lisbon tiles"}
    assert [p.user_id for p in personal.data.profiles] == ["u_chen"]
    assert "Chen likes Lisbon tiles" not in _texts(group)
    assert group.data.profiles == []


async def test_several_groups_form_one_pool(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        pooled = await world.memory.search(
            _search(group_ids=["trip", "ops"], asker_id="u_chen")
        )
        evan = await world.memory.search(_search(group_ids=["trip"], asker_id="u_evan"))
    finally:
        await cm.__aexit__(None, None, None)

    assert {e.group_id for e in pooled.data.episodes} == {"trip", "ops"}
    scores = [e.score for e in pooled.data.episodes]
    assert scores == sorted(scores, reverse=True)
    assert _texts(evan) == {_LATE}


async def test_get_lists_only_the_askers_window(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        chen = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="episode",
            )
        )
        evan = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_evan",
                memory_type="episode",
            )
        )
    finally:
        await cm.__aexit__(None, None, None)

    assert {e.episode for e in chen.data.episodes} == {_EARLY, _LATE}
    assert [e.episode for e in evan.data.episodes] == [_LATE]
    assert evan.data.total_count == 1
    assert {e.group_id for e in chen.data.episodes} == {"trip"}


def test_a_group_get_lists_episodes_only() -> None:
    with pytest.raises(ValueError, match="group scope"):
        GetRequest(group_id="trip", asker_id="u_chen", memory_type="profile")


async def test_a_referenced_picture_is_attached_only_when_visible(
    tmp_path: Path,
) -> None:
    world, cm = await _world(tmp_path)
    try:
        # A picture posted before Evan joined, referred to by a later episode.
        early = await world.picture(
            world.scope("trip"), "Lisbon venue floor plan", at=1_500, shade=10
        )
        await world.episode(
            "trip",
            "The Lisbon offsite plan refers to the venue floor plan",
            start=7_500,
            end=7_600,
            image_ids=[early],
        )
        await world.project()
        chen = await world.memory.search(
            _search(group_id="trip", asker_id="u_chen", top_k=10)
        )
        evan = await world.memory.search(
            _search(group_id="trip", asker_id="u_evan", top_k=10)
        )
    finally:
        await cm.__aexit__(None, None, None)

    def attached(response: Any) -> set[str]:
        return {
            image.image_id
            for episode in response.data.episodes
            for image in episode.images
        } | {image.image_id for image in response.data.images}

    assert early in attached(chen)
    assert early not in attached(evan)
