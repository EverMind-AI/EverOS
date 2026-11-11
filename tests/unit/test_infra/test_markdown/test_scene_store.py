"""A user's scene versions in markdown (docs/scenes.md), against a real root.

What these pin:

- a user's versions live under ``users/<u>/scenes/scene-<date>.md`` with
  ``sc_`` ids, read back whole; a new version marks the one it supersedes;
- a forget marks a version and an undo clears the mark, file by file;
- a deletion removes every version that names one of its episodes, matched
  on the parsed member list, never on a text prefix;
- an agent has no scenes;
- a whole-owner deletion lists the scene versions with the episode table's
  refs, so their rows go with the rest;
- the cascade matches the user scene files to the ``scene`` kind, which
  projects them into the episode table as ``parent_type = scene``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.cascade import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownEpisodeStore,
    MarkdownOwnerDirStore,
    MarkdownSceneStore,
)
from everos.memory.models import FORGOTTEN_MARK, Episode, Scene
from everos.memory.ports import NO_SOURCE_CHECK, Scope

_TS = 1_747_216_800_000


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _user() -> Scope:
    return Scope(app_id="app", project_id="p", owner_id="u1", owner_type="user")


def _scene(members: list[str], *, version: int = 1, text: str = "story") -> Scene:
    return Scene(
        owner_id="u1",
        scene_id="cl_abc",
        version=version,
        member_ids=members,
        subject="Lisbon trip",
        summary="Where it stands.",
        narrative=text,
        timestamp=_TS + 10,
        source_start=_TS,
        sender_ids=["u1"],
        language="English",
    )


async def test_versions_round_trip_and_supersede(root: MemoryRoot) -> None:
    store = MarkdownSceneStore(root)
    first = await store.append(
        _user(), _scene(["ep_1", "ep_2"]), supersedes=None, only_if=NO_SOURCE_CHECK
    )
    assert first is not None and first.entry_id.startswith("sc_")
    assert first.md_path.startswith("app/p/users/u1/scenes/scene-")
    v1 = await store.latest(_user(), "cl_abc")
    assert v1 is not None and v1.item == _scene(["ep_1", "ep_2"])

    second = await store.append(
        _user(),
        _scene(["ep_1", "ep_2", "ep_3"], version=2, text="longer story"),
        supersedes=v1,
        only_if=NO_SOURCE_CHECK,
    )
    assert second is not None
    latest = await store.latest(_user(), "cl_abc")
    assert latest is not None and latest.item.version == 2
    assert await store.deprecations(_user(), first.md_path) == {
        first.entry_id: second.entry_id
    }
    assert [v.item.version for v in await store.versions(_user())] == [1, 2]
    assert match_kind(second.md_path).name == "scene"  # type: ignore[union-attr]


async def test_a_forget_mark_and_its_undo(root: MemoryRoot) -> None:
    store = MarkdownSceneStore(root)
    ref = await store.append(
        _user(), _scene(["ep_1"]), supersedes=None, only_if=NO_SOURCE_CHECK
    )
    assert ref is not None
    await store.deprecate(
        _user(),
        entries=[(ref.md_path, ref.entry_id)],
        replaced_by=FORGOTTEN_MARK,
        only_if=NO_SOURCE_CHECK,
    )
    assert await store.deprecations(_user(), ref.md_path) == {
        ref.entry_id: FORGOTTEN_MARK
    }
    await store.clear_deprecations(_user(), [ref.entry_id])
    assert await store.deprecations(_user(), ref.md_path) == {}


async def test_a_deletion_removes_the_versions_naming_its_episodes(
    root: MemoryRoot,
) -> None:
    store = MarkdownSceneStore(root)
    await store.append(
        _user(), _scene(["ep_1", "ep_2"]), supersedes=None, only_if=NO_SOURCE_CHECK
    )
    keep = await store.append(
        _user(),
        _scene(["ep_10", "ep_20"]).model_copy(update={"scene_id": "cl_other"}),
        supersedes=None,
        only_if=NO_SOURCE_CHECK,
    )
    removed = await store.remove_naming(_user(), ["ep_2"])
    assert len(removed) == 1
    # "ep_1" is a prefix of "ep_10": the member list decides, not the text.
    assert await store.remove_naming(_user(), ["ep_1"]) == []
    assert [v.ref.entry_id for v in await store.versions(_user())] == [
        keep.entry_id  # type: ignore[union-attr]
    ]


async def test_an_agent_has_no_scenes(root: MemoryRoot) -> None:
    agent = Scope(app_id="app", project_id="p", owner_id="a1", owner_type="agent")
    with pytest.raises(ValueError, match="user or a group"):
        await MarkdownSceneStore(root).append(
            agent, _scene(["ac_1"]), supersedes=None, only_if=NO_SOURCE_CHECK
        )


async def test_an_owner_deletion_lists_the_scenes_with_the_episodes(
    root: MemoryRoot,
) -> None:
    (episode,) = await MarkdownEpisodeStore(root).append_many(
        _user(),
        [
            Episode(
                owner_id="u1",
                episode="e",
                summary="e",
                timestamp=_TS,
                session_id="s1",
                parent_id="mc_1",
            )
        ],
        only_if=NO_SOURCE_CHECK,
    )
    scene = await MarkdownSceneStore(root).append(
        _user(), _scene([episode.entry_id]), supersedes=None, only_if=NO_SOURCE_CHECK
    )
    deletion = await MarkdownOwnerDirStore(root).delete(_user())
    assert sorted(r.entry_id for r in deletion.refs["episode"]) == sorted(
        [episode.entry_id, scene.entry_id]  # type: ignore[union-attr]
    )
    assert deletion.count == 2
