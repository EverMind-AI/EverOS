"""A user's scenes through the service (docs/scenes.md).

What these pin:

- **Search competition.** With ``[scenes] enabled``, two conversations about
  one matter make a scene under ``users/<u>/scenes/``; the cascade projects
  it into the episode table, and a search returns it next to the episodes,
  as ``type = Scene``, from the same ranking. ``get`` lists it under
  ``memory_type = scene`` and never under ``episode``.
- **The switch.** Off by default: no scene file, no scene call.
- **Forgetting follows the sources.** Forgetting one of a scene's
  conversations keeps the scene; forgetting the last one marks it forgotten
  and erases its row; undoing brings it back.
- **Deletion.** Deleting a conversation removes every scene version that
  names one of its episodes.
- **A picture forget** writes the scene again without the picture's
  reference, as a scene: no scene row is rewritten as an episode.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config import MemorizeSettings, Settings
from everos.memory.models import FORGOTTEN_MARK, Episode, Scene
from everos.memory.ports import MemoryRef, NoSourceCheck, Scope, Stored
from everos.memory.strategies import UPDATE_USER_SCENES_SPEC
from everos.service import GetRequest, SearchRequest
from everos.service.memory._delete import Deleter
from everos.service.memory._forget_session import SessionForgetManager
from everos.service.memory._undo import UndoManager
from tests.fakes import make_stores
from tests.helpers.groups import APP, PROJECT, GroupWorld, png

T0 = 1_772_323_200_000
_TEXT = "Amy planned the Lisbon trip and booked a flat in Alfama"


@pytest.fixture
def scenes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """``[scenes] enabled = true`` for the strategy under test (default off)."""
    base = Settings()
    settings = base.model_copy(
        update={"scenes": base.scenes.model_copy(update={"enabled": True})}
    )
    module = importlib.import_module("everos.memory.strategies.update_user_scenes")
    monkeypatch.setattr(module, "load_settings", lambda: settings)


def _model(narrative: str = "Scene narrative: Amy's Lisbon trip took shape.") -> Any:
    def answer(messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "_EpisodeReply":
            # One text for every segment: equal embeddings, so one scene.
            return json.dumps({"title": "Lisbon", "summary": _TEXT, "content": _TEXT})
        if name == "SceneReply":
            return json.dumps(
                {
                    "narrative": narrative,
                    "summary": "Amy's Lisbon trip.",
                    "title": "Lisbon trip",
                }
            )
        raise AssertionError(f"unexpected model call {name}")

    return answer


async def _conversation(world: GroupWorld, session_id: str, at: int) -> None:
    await world.memory.add(
        {
            "session_id": session_id,
            "app_id": APP,
            "project_id": PROJECT,
            "messages": [
                {"sender_id": "amy", "role": "user", "timestamp": T0 + at,
                 "content": "I booked the flat in Alfama for Lisbon."},
                {"sender_id": "bot", "role": "assistant", "timestamp": T0 + at + 500,
                 "content": "Great, Alfama is lovely."},
            ],
        },
        request_id=f"r_{session_id}",
        async_mode=False,
        defer_extraction=True,
    )  # fmt: skip
    await world.memory.flush(
        session_id=session_id, request_id=f"f_{session_id}", app_id=APP,
        project_id=PROJECT,
    )  # fmt: skip
    await world.run(UPDATE_USER_SCENES_SPEC)


def _scenes_dir(tmp_path: Path) -> Path:
    return tmp_path / "default_app" / PROJECT / "users" / "amy" / "scenes"


@pytest.mark.usefixtures("scenes_on")
async def test_a_scene_competes_with_the_episodes_in_search(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_model())) as world:
        await _conversation(world, "s1", 1_000)
        await _conversation(world, "s2", 9_000)
        assert len(list(_scenes_dir(tmp_path).glob("scene-*.md"))) == 1
        await world.project()

        found = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="amy",
                query="Lisbon trip Alfama flat",
                top_k=20,
            )
        )
        assert sorted(e.type for e in found.data.episodes) == [
            "Conversation",
            "Conversation",
            "Scene",
        ]
        (scene,) = [e for e in found.data.episodes if e.type == "Scene"]
        assert scene.user_id == "amy" and scene.episode.startswith("Scene narrative")

        listed = await world.memory.get(
            GetRequest(
                app_id=APP, project_id=PROJECT, user_id="amy", memory_type="scene"
            )
        )
        assert [s.type for s in listed.data.scenes] == ["Scene"]
        episodes = await world.memory.get(
            GetRequest(
                app_id=APP, project_id=PROJECT, user_id="amy", memory_type="episode"
            )
        )
        assert {e.type for e in episodes.data.episodes} == {"Conversation"}
        assert episodes.data.total_count == 2


async def test_the_default_writes_no_scene_and_makes_no_call(tmp_path: Path) -> None:
    assert Settings().scenes.enabled is False
    llm = FakeLLMClient(handler=_model())
    async with GroupWorld.open(tmp_path, llm=llm) as world:
        await _conversation(world, "s1", 1_000)
        await _conversation(world, "s2", 9_000)
        assert not _scenes_dir(tmp_path).exists()
        extracted = [
            e for e in world.bus.events if type(e).__name__ == "EpisodeExtracted"
        ]
        assert len(extracted) == 2  # the strategy had its events, and skipped them
    assert "SceneReply" not in [call.response_format.__name__ for call in llm.calls]


@pytest.mark.usefixtures("scenes_on")
async def test_a_picture_forget_rewrites_the_scene_as_a_scene(tmp_path: Path) -> None:
    image_id = hashlib.sha256(png(10)).hexdigest()  # the picture's content hash
    model = _model(f"Amy sent a photo (img-1: {image_id}) of the Alfama flat.")
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        stored = await world.picture(
            world.personal("amy"), "the Alfama flat", at=T0, shade=10
        )
        assert stored == image_id
        await _conversation(world, "s1", 1_000)
        await _conversation(world, "s2", 9_000)
        await world.project()

        await world.memory.forget_image(
            app_id=APP, project_id=PROJECT, user_id="amy", image_id=image_id
        )
        scenes = await world.stores.scene.versions(world.personal("amy"))
        assert [s.item.version for s in scenes] == [1, 2]
        assert image_id in scenes[0].item.narrative
        assert image_id not in scenes[1].item.narrative
        assert scenes[1].item.narrative.startswith("Amy sent a photo of the")
        episodes = (tmp_path / "default_app" / PROJECT / "users" / "amy").glob(
            "episodes/*.md"
        )
        assert all("**parent_type**: scene" not in p.read_text() for p in episodes)


# ── forget, undo and delete on the stores ────────────────────────────────


_SCOPE = Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


class _Eraser:
    def __init__(self) -> None:
        self.erased: list[dict[str, list[MemoryRef]]] = []
        self.requeued: list[tuple[str, set[str]]] = []

    async def erase(self, scope: Scope, removed: dict[str, list[MemoryRef]]) -> None:
        self.erased.append(dict(removed))

    async def erase_owner(self, scope: Scope, kinds: Any) -> None:
        return None

    async def requeue(self, kind: str, md_paths: Any) -> None:
        self.requeued.append((kind, set(md_paths)))


async def _world() -> tuple[Any, list[str], Stored[Scene]]:
    """Two sessions' episodes and one scene over both."""
    stores = make_stores()
    ids = []
    for n, session in enumerate(("s1", "s2"), 1):
        ref = await stores.episode.append(
            _SCOPE,
            Episode(
                owner_id="u1",
                episode=f"episode {n}",
                summary="s",
                timestamp=1_000 * n,
                session_id=session,
                parent_id=f"mc_{n}",
            ),
            only_if=NoSourceCheck(),
        )
        assert ref is not None
        ids.append(ref.entry_id)
    await stores.scene.append(
        _SCOPE,
        Scene(
            owner_id="u1",
            scene_id="cl_1",
            version=1,
            member_ids=ids,
            subject="t",
            summary="s",
            narrative="both episodes",
            timestamp=2_000,
            source_start=1_000,
        ),
        supersedes=None,
        only_if=NoSourceCheck(),
    )
    (scene,) = stores.scene.written
    return stores, ids, scene


async def test_forgetting_follows_the_sources_and_undo_brings_it_back() -> None:
    stores, _ids, scene = await _world()
    eraser = _Eraser()
    forget = SessionForgetManager(stores=stores, images=stores.images, eraser=eraser)
    undo = UndoManager(stores=stores, images=stores.images, eraser=eraser)

    first = await forget.forget_session(scope=_SCOPE, session_id="s1")
    assert first.scenes == [] and stores.scene.current(_SCOPE) == [scene]

    second = await forget.forget_session(scope=_SCOPE, session_id="s2")
    assert second.scenes == [scene.ref.entry_id]
    assert stores.scene.superseded == {scene.ref.entry_id: FORGOTTEN_MARK}
    assert scene.ref in eraser.erased[-1]["episode"]  # its row goes at once

    restored = await undo.undo_session(scope=_SCOPE, session_id="s2")
    assert restored.applied
    assert stores.scene.superseded == {}
    assert ("scene", {scene.ref.md_path}) in eraser.requeued


async def test_deleting_a_conversation_removes_the_scenes_naming_it() -> None:
    stores, _ids, scene = await _world()
    eraser = _Eraser()
    deleter = Deleter(stores=stores, eraser=eraser, settings=MemorizeSettings())
    await deleter.delete(
        app_id="app", project_id="proj", user_id="u1", agent_id=None, session_id="s2"
    )
    assert stores.scene.written == []
    assert scene.ref in eraser.erased[-1]["episode"]
