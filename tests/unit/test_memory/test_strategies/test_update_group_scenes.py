"""update_group_scenes: the group's scene versions (docs/group-memory.md, 7.7).

In-memory stores, a fixed-vector embedder and a scripted reflector. What
these pin:

- the second episode of a cluster writes version 1 (INIT over both); a
  third writes version 2 (UPDATE with the new one) superseding version 1;
- a version spans its members and unites their senders and subjects;
  members of one origin keep it, mixed ones read ``mixed``;
- a replay writes nothing new, and a retry after a failed narrative call
  still writes the version (the episode was placed before the call failed);
- the narrative is written in the language the event names (the group
  pipeline's resolved language), on INIT and on UPDATE;
- without an embedder the strategy does nothing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.memory.events import GroupEpisodeExtracted
from everos.memory.models import Episode
from everos.memory.ports import NoSourceCheck, Scope
from everos.memory.strategies import update_group_scenes
from tests.fakes import make_deps

_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")


class _Same:
    """Every text embeds to one vector: every episode joins one scene."""

    dim = 4

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)


class _Reflector:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_first = fail_first

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        assert response_format.__name__ == "_ReflectOutput"
        self.calls.append("\n".join(str(m.content) for m in messages))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("narrative call failed")
        n = len(self.calls)
        return json.dumps(
            {"title": f"Scene {n}", "summary": f"v{n}", "content": f"Narrative {n}"}
        )


async def _episode(deps: Any, n: int, *, origin: str, sender: str) -> str:
    ref = await deps.stores.episode.append(
        _SCOPE,
        Episode(
            owner_id="trip",
            episode=f"Episode {n} about the offsite",
            summary="s",
            subject="Offsite",
            timestamp=1_000 * n + 500,
            source_start=1_000 * n,
            session_id="s1",
            sender_ids=[sender],
            subject_ids=[sender],
            origin=origin,  # type: ignore[arg-type]
            parent_id=f"mc_{n}",
        ),
        only_if=NoSourceCheck(),
    )
    return ref.entry_id


def _event(
    entry_id: str, n: int, *, language: str | None = None
) -> GroupEpisodeExtracted:
    return GroupEpisodeExtracted(
        memcell_id=f"mc_{n}",
        episode_entry_id=entry_id,
        episode_text=f"Episode {n} about the offsite",
        episode_timestamp_ms=1_000 * n + 500,
        group_id="trip",
        app_id="default",
        project_id="p",
        language=language,
    )


async def test_versions_span_their_members_and_supersede() -> None:
    reflector = _Reflector()
    deps = make_deps(llm=FakeLLMClient(handler=reflector), embedding=_Same())
    first = await _episode(deps, 1, origin="member", sender="u_chen")
    await update_group_scenes(_event(first, 1), deps)
    assert deps.stores.scene.versions == []  # one episode is not a scene

    second = await _episode(deps, 2, origin="assistant", sender="bot_ava")
    await update_group_scenes(_event(second, 2), deps)
    (v1,) = deps.stores.scene.versions
    assert v1.item.version == 1 and v1.item.narrative == "Narrative 1"
    assert v1.item.member_ids == [first, second]
    assert (v1.item.source_start, v1.item.timestamp) == (1_000, 2_500)
    assert v1.item.sender_ids == ["u_chen", "bot_ava"]
    assert v1.item.origin == "mixed"

    third = await _episode(deps, 3, origin="member", sender="u_betty")
    await update_group_scenes(_event(third, 3), deps)
    v2 = deps.stores.scene.versions[-1]
    assert v2.item.version == 2 and v2.item.scene_id == v1.item.scene_id
    assert deps.stores.scene.superseded == {v1.ref.entry_id: v2.ref.entry_id}
    # UPDATE mode: the old narrative and only the new episode.
    assert "Narrative 1" in reflector.calls[-1]
    assert "Episode 3" in reflector.calls[-1]
    assert "Episode 1" not in reflector.calls[-1]

    await update_group_scenes(_event(third, 3), deps)  # a replay
    assert len(deps.stores.scene.versions) == 2


async def test_a_retry_after_a_failed_narrative_writes_the_version() -> None:
    deps = make_deps(
        llm=FakeLLMClient(handler=_Reflector(fail_first=True)), embedding=_Same()
    )
    first = await _episode(deps, 1, origin="member", sender="u_chen")
    await update_group_scenes(_event(first, 1), deps)
    second = await _episode(deps, 2, origin="member", sender="u_chen")
    with pytest.raises(RuntimeError):
        await update_group_scenes(_event(second, 2), deps)
    assert deps.stores.scene.versions == []
    await update_group_scenes(_event(second, 2), deps)  # the engine's retry
    (v1,) = deps.stores.scene.versions
    assert v1.item.member_ids == [first, second]
    assert v1.item.origin == "member"


async def test_the_narrative_is_written_in_the_event_s_language() -> None:
    reflector = _Reflector()
    deps = make_deps(llm=FakeLLMClient(handler=reflector), embedding=_Same())
    for n in (1, 2, 3):
        entry = await _episode(deps, n, origin="member", sender="u_chen")
        await update_group_scenes(_event(entry, n, language="English"), deps)

    assert len(reflector.calls) == 2  # INIT, then UPDATE
    for prompt in reflector.calls:
        assert "Write every output field in English" in prompt
    assert {v.item.language for v in deps.stores.scene.versions} == {"English"}


async def test_without_an_embedder_nothing_happens() -> None:
    deps = make_deps(embedding=None)
    first = await _episode(deps, 1, origin="member", sender="u_chen")
    assert await update_group_scenes(_event(first, 1), deps) == []
    assert deps.stores.scene.versions == []
