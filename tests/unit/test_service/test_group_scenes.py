"""Group scenes (docs/group-memory.md, section 7.7; task 6).

The bootstrap's memory service writes the group's segments, the
``update_group_scenes`` strategy runs on the events it published, and the
cascade projects the markdown into a real LanceDB. What these pin:

- two episodes about one thing make a scene, written under
  ``groups/<g>/scenes/``; a third makes a new version that supersedes the
  first, and only the newest competes;
- the scene is withheld from an asker who cannot see all of its episodes --
  a member who left and rejoined between them -- while each episode still
  returns on its own; an asker who sees them all sees the scene;
- get lists scenes under ``memory_type = scene``, and ``episode`` lists no
  scene;
- scenes are an opt-in: with the default ``[groups] scenes = false`` no
  scene is written and no narrative call is made. The tests of what a scene
  does switch them on.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config import Settings
from everos.memory.strategies import UPDATE_GROUP_SCENES_SPEC
from everos.service import GetRequest, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000
_TEXT = "Chen and Dan planned the Lisbon offsite venue"


@pytest.fixture
def scenes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """``[groups] scenes = true`` for the strategy under test (default off)."""
    base = Settings()
    settings = base.model_copy(
        update={"groups": base.groups.model_copy(update={"scenes": True})}
    )
    # The package re-exports the function under the module's name.
    module = importlib.import_module("everos.memory.strategies.update_group_scenes")
    monkeypatch.setattr(module, "load_settings", lambda: settings)


def _model(messages: Any, *, response_format: type, **_: Any) -> str:
    name = response_format.__name__
    if name == "_BatchReply":
        return json.dumps({"reasoning": "s", "boundaries": [], "tail_complete": False})
    if name == "EpisodeReply":
        # One text for every segment: equal embeddings, so one scene.
        return json.dumps({"title": "Offsite", "summary": _TEXT, "content": _TEXT})
    if name == "_ReflectOutput":
        return json.dumps(
            {
                "title": "Lisbon offsite scene",
                "summary": "How the Lisbon offsite took shape.",
                "content": "Scene narrative: the Lisbon offsite took shape.",
            }
        )
    raise AssertionError(f"unexpected model call {name}")


async def _segment(world: GroupWorld, at: int) -> None:
    await world.memory.add(
        {
            "session_id": "s_group",
            "app_id": APP,
            "project_id": PROJECT,
            "group_id": "trip",
            "messages": [
                {"sender_id": "u_chen", "role": "user", "timestamp": T0 + at,
                 "content": "Lisbon offsite venue?"},
                {"sender_id": "u_chen", "role": "user", "timestamp": T0 + at + 500,
                 "content": "Let's book it."},
            ],
        },
        request_id="r",
        async_mode=False,
        defer_extraction=True,
    )  # fmt: skip
    await world.memory.flush(
        session_id="s_group", request_id="f", app_id=APP, project_id=PROJECT
    )
    await world.run(UPDATE_GROUP_SCENES_SPEC)


async def _setup(world: GroupWorld) -> None:
    await world.roster(
        "trip",
        [
            join("u_chen", T0, display_name="Chen"),
            join("u_dan", T0, display_name="Dan"),
            leave("u_dan", T0 + 2_500),
            join("u_dan", T0 + 7_500),
        ],
    )


def _search(asker: str) -> SearchRequest:
    return SearchRequest(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=asker,
        query="Lisbon offsite venue",
        top_k=20,
    )


@pytest.mark.usefixtures("scenes_on")
async def test_a_scene_is_withheld_unless_all_its_episodes_are_visible(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_model)) as world:
        await _setup(world)
        await _segment(world, 1_000)  # inside Dan's first window
        await _segment(world, 8_000)  # inside Dan's second window
        scenes = tmp_path / "default_app" / PROJECT / "groups" / "trip" / "scenes"
        assert len(list(scenes.glob("scene-*.md"))) == 1
        await world.project()

        chen = await world.memory.search(_search("u_chen"))
        dan = await world.memory.search(_search("u_dan"))
        assert sorted(e.type for e in chen.data.episodes) == [
            "Conversation",
            "Conversation",
            "Scene",
        ]
        # Dan sees each episode, but not the scene spanning his absence.
        assert sorted(e.type for e in dan.data.episodes) == [
            "Conversation",
            "Conversation",
        ]

        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="scene",
            )
        )
        (scene,) = listed.data.scenes
        assert scene.type == "Scene" and scene.group_id == "trip"
        assert scene.episode.startswith("Scene narrative")
        episodes = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="episode",
            )
        )
        assert {e.type for e in episodes.data.episodes} == {"Conversation"}


@pytest.mark.usefixtures("scenes_on")
async def test_a_new_member_episode_supersedes_the_scene_version(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_model)) as world:
        await _setup(world)
        for at in (1_000, 1_500, 2_000):
            await _segment(world, at * 2)
        await world.project()
        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="scene",
            )
        )
        assert len(listed.data.scenes) == 1  # version 2; version 1 superseded
        files = list(
            (tmp_path / "default_app" / PROJECT / "groups" / "trip" / "scenes").glob(
                "*.md"
            )
        )
        text = files[0].read_text()
        assert "**version**: 1" in text and "**version**: 2" in text
        assert "deprecated_entries:" in text


async def test_the_default_writes_no_scene(tmp_path: Path) -> None:
    """Off by default: two episodes of one scene, and neither a scene nor a call."""
    assert Settings().groups.scenes is False
    llm = FakeLLMClient(handler=_model)
    async with GroupWorld.open(tmp_path, llm=llm) as world:
        await _setup(world)
        await _segment(world, 1_000)
        await _segment(world, 2_000)
        scenes = tmp_path / "default_app" / PROJECT / "groups" / "trip" / "scenes"
        assert not scenes.exists()
        episodes = [
            e for e in world.bus.events if type(e).__name__ == "GroupEpisodeExtracted"
        ]
        assert len(episodes) == 2  # the strategy had its events, and skipped them
    called = [call.response_format.__name__ for call in llm.calls]
    assert "_ReflectOutput" not in called
