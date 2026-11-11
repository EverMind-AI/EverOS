"""update_user_scenes: a user's scene versions (docs/scenes.md).

In-memory stores, a fixed-vector embedder and a scripted scene model. What
these pin:

- an episode that joins no scene costs no call; the second episode of a
  cluster writes version 1 over both (empty current narrative); a third
  writes version 2 from version 1's narrative and the new episode only,
  superseding version 1; a replay writes nothing;
- the call is capped: the system prompt asks for ``narrative_max_words``,
  the request carries the token cap with ``retry_cut``, a reply past the cap
  is clipped at a sentence end, so the next prompt stays bounded, and at most
  ``max_new_episodes`` episodes (each clipped) are read;
- ``min_members`` delays the first version;
- forgetting follows the sources: a forgotten member is no member, a version
  written from an episode forgotten since is not carried forward, and a
  forgotten version keeps its mark when the scene goes on;
- the switch: off by default, and then the strategy does not apply.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config import Settings
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.scene import (
    SCENE_UPDATE,
    SceneCaps,
    narrative_char_cap,
    reply_token_cap,
    write_narrative,
)
from everos.memory.models import FORGOTTEN_MARK, Episode
from everos.memory.ports import NoSourceCheck, Scope
from everos.memory.strategies import UPDATE_USER_SCENES_SPEC, update_user_scenes
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="amy", owner_type="user")
_EMPTY = '<current_narrative words="0">\n\n</current_narrative>'
"""A new scene's prompt: an empty current narrative."""


class _Same:
    """Every text embeds to one vector: every episode joins one scene."""

    dim = 4

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)


class _SceneModel:
    """Answers the scene call; records each request's text and arguments."""

    def __init__(self, narrative: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.narrative = narrative

    def __call__(self, messages: Any, *, response_format: type, **kw: Any) -> str:
        assert response_format.__name__ == "SceneReply"
        self.calls.append(
            {
                "system": str(messages[0].content),
                "user": str(messages[1].content),
                **kw,
            }
        )
        n = len(self.calls)
        return json.dumps(
            {
                "narrative": self.narrative or f"Narrative {n}.",
                "summary": f"Where it stands {n}.",
                "title": f"Scene {n}",
            }
        )


@pytest.fixture
def scenes(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Turn ``[scenes]`` on for the strategy; returns a setter for its fields."""
    module = importlib.import_module("everos.memory.strategies.update_user_scenes")

    def configure(**fields: Any) -> Settings:
        base = Settings()
        settings = base.model_copy(
            update={
                "scenes": base.scenes.model_copy(update={"enabled": True, **fields})
            }
        )
        monkeypatch.setattr(module, "load_settings", lambda: settings)
        return settings

    configure()
    return configure


async def _episode(deps: Any, n: int, *, text: str | None = None) -> str:
    ref = await deps.stores.episode.append(
        _SCOPE,
        Episode(
            owner_id="amy",
            episode=text or f"Episode {n}: Amy planned the Lisbon trip.",
            summary="s",
            subject="Lisbon",
            timestamp=1_000 * n,
            session_id=f"s{n}",
            sender_ids=["amy"],
            parent_id=f"mc_{n}",
        ),
        only_if=NoSourceCheck(),
    )
    return ref.entry_id


def _event(entry_id: str, n: int, *, text: str | None = None) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=f"mc_{n}",
        episode_entry_id=entry_id,
        episode_text=text or f"Episode {n}: Amy planned the Lisbon trip.",
        episode_timestamp_ms=1_000 * n,
        owner_id="amy",
        session_id=f"s{n}",
        app_id="default",
        project_id="p",
        language="English",
    )


async def _add(deps: Any, n: int, *, text: str | None = None) -> str:
    entry_id = await _episode(deps, n, text=text)
    await update_user_scenes(_event(entry_id, n, text=text), deps)
    return entry_id


@pytest.mark.usefixtures("scenes")
async def test_a_scene_forms_at_the_second_episode_and_versions_supersede() -> None:
    model = _SceneModel()
    deps = make_deps(llm=FakeLLMClient(handler=model), embedding=_Same())
    first = await _add(deps, 1)
    assert deps.stores.scene.written == [] and model.calls == []  # joins no scene

    second = await _add(deps, 2)
    (v1,) = deps.stores.scene.written
    assert v1.item.version == 1 and v1.item.member_ids == [first, second]
    assert v1.item.narrative == "Narrative 1." and v1.item.subject == "Scene 1"
    assert v1.ref.entry_id.startswith("sc_")
    assert _EMPTY in model.calls[0]["user"]
    assert "1. Episode 1" in model.calls[0]["user"]
    assert "2. Episode 2" in model.calls[0]["user"]

    third = await _add(deps, 3)
    v2 = deps.stores.scene.written[-1]
    assert v2.item.version == 2 and v2.item.scene_id == v1.item.scene_id
    assert v2.item.member_ids == [first, second, third]
    assert deps.stores.scene.superseded == {v1.ref.entry_id: v2.ref.entry_id}
    # The update reads the current narrative and only the joining episode.
    assert "Narrative 1." in model.calls[1]["user"]
    assert "Episode 3" in model.calls[1]["user"]
    assert "Episode 1" not in model.calls[1]["user"]

    await update_user_scenes(_event(third, 3), deps)  # a replay
    assert len(deps.stores.scene.written) == 2 and len(model.calls) == 2


async def test_the_call_is_capped_and_a_long_reply_is_clipped(scenes: Any) -> None:
    scenes(narrative_max_words=60)
    sentence = "Amy booked a flat in Alfama on 3 May 2023. "
    closing = "Now the trip is set for June."
    model = _SceneModel(narrative=sentence * 200 + closing)  # far past the cap
    deps = make_deps(llm=FakeLLMClient(handler=model), embedding=_Same())
    await _add(deps, 1)
    await _add(deps, 2)
    (v1,) = deps.stores.scene.written
    cap = narrative_char_cap(60)
    assert len(v1.item.narrative) <= cap
    # The oldest part goes, at a sentence start; the closing state stays.
    assert v1.item.narrative.startswith("Amy booked")
    assert v1.item.narrative.endswith(closing)
    call = model.calls[0]
    assert "At most 60 words in all" in call["system"]
    assert call["max_tokens"] == reply_token_cap(60)
    assert call["retry_cut"] is True

    await _add(deps, 3)
    # The next prompt carries the clipped narrative, not the reply.
    assert len(model.calls[1]["user"]) < cap + 1_000


async def test_one_call_reads_the_newest_episodes_each_clipped() -> None:
    model = _SceneModel()
    episodes = [f"Episode {n}: " + "word " * 2_000 for n in range(1, 7)]
    await write_narrative(
        FakeLLMClient(handler=model),
        previous="",
        episodes=episodes,
        caps=SceneCaps(
            narrative_max_words=300, max_new_episodes=4, episode_max_chars=500
        ),
        language="English",
        custom_instructions=None,
    )
    user = model.calls[0]["user"]
    assert "Episode 1:" not in user and "Episode 2:" not in user
    assert all(f"Episode {n}:" in user for n in (3, 4, 5, 6))
    assert len(user) < 4 * 500 + 400


async def test_min_members_delays_the_first_version(scenes: Any) -> None:
    scenes(min_members=3)
    model = _SceneModel()
    deps = make_deps(llm=FakeLLMClient(handler=model), embedding=_Same())
    first = await _add(deps, 1)
    second = await _add(deps, 2)
    assert deps.stores.scene.written == [] and model.calls == []
    third = await _add(deps, 3)
    (v1,) = deps.stores.scene.written
    assert v1.item.member_ids == [first, second, third]
    assert len(model.calls) == 1


@pytest.mark.usefixtures("scenes")
async def test_a_forgotten_member_is_dropped_and_its_narrative_not_carried() -> None:
    model = _SceneModel()
    deps = make_deps(llm=FakeLLMClient(handler=model), embedding=_Same())
    first = await _add(deps, 1, text="Episode 1: Amy's secret plan.")
    second = await _add(deps, 2)
    (v1,) = deps.stores.scene.written
    # Episode 1 is forgotten (a session forget marks it).
    md_path = (await deps.stores.episode.read_many(_SCOPE, [first]))[0].ref.md_path
    await deps.stores.episode.deprecate(
        _SCOPE,
        entries=[(md_path, first)],
        replaced_by=FORGOTTEN_MARK,
        only_if=NoSourceCheck(),
    )
    third = await _add(deps, 3)
    v2 = deps.stores.scene.written[-1]
    assert v2.item.member_ids == [second, third]
    # Written from the live members alone: no carried narrative, no secret.
    assert _EMPTY in model.calls[1]["user"]
    assert "secret" not in model.calls[1]["user"]
    assert "Episode 2" in model.calls[1]["user"]
    assert deps.stores.scene.superseded == {v1.ref.entry_id: v2.ref.entry_id}


@pytest.mark.usefixtures("scenes")
async def test_a_forgotten_version_keeps_its_mark_when_the_scene_goes_on() -> None:
    model = _SceneModel()
    deps = make_deps(llm=FakeLLMClient(handler=model), embedding=_Same())
    await _add(deps, 1)
    await _add(deps, 2)
    (v1,) = deps.stores.scene.written
    deps.stores.scene.superseded[v1.ref.entry_id] = FORGOTTEN_MARK
    await _add(deps, 3)
    v2 = deps.stores.scene.written[-1]
    assert v2.item.version == 2
    assert deps.stores.scene.superseded[v1.ref.entry_id] == FORGOTTEN_MARK
    assert "Narrative 1." not in model.calls[1]["user"]


async def test_the_switch_is_off_by_default() -> None:
    assert Settings().scenes.enabled is False
    deps = make_deps(llm=FakeLLMClient(handler=_SceneModel()), embedding=_Same())
    event = _event("ep_20260101_0001", 1)
    assert UPDATE_USER_SCENES_SPEC.applies_to is not None
    assert UPDATE_USER_SCENES_SPEC.applies_to(event, deps) is False


@pytest.mark.usefixtures("scenes")
async def test_only_pipeline_episodes_with_an_embedder_apply() -> None:
    deps = make_deps(llm=FakeLLMClient(handler=_SceneModel()), embedding=_Same())
    event = _event("ep_20260101_0001", 1)
    assert UPDATE_USER_SCENES_SPEC.applies_to(event, deps) is True
    merged = event.model_copy(update={"source": "reflection"})
    assert UPDATE_USER_SCENES_SPEC.applies_to(merged, deps) is False
    bare = make_deps(llm=FakeLLMClient(handler=_SceneModel()), embedding=None)
    assert UPDATE_USER_SCENES_SPEC.applies_to(event, bare) is False


def test_the_prompt_is_registered_and_pinned() -> None:
    assert SCENE_UPDATE.id == "chat.scene_update"
    assert SCENE_UPDATE.version == 2
    assert SCENE_UPDATE.fingerprint == "6ce8a6f4593bf434"
