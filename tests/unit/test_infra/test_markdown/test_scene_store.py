"""A user's scene versions in markdown (docs/scenes.md), against a real root.

What these pin:

- a user's versions live under ``users/<u>/scenes/scene-<date>.md`` with
  ``sc_`` ids, read back whole; a new version marks the one it supersedes;
- a forget marks a version and an undo clears the mark, file by file;
- an agent has no scenes;
- the cascade matches the user scene files to the ``scene`` kind, which
  projects them into the episode table as ``parent_type = scene``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.cascade import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownSceneStore
from everos.memory.models import FORGOTTEN_MARK, Scene
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


async def test_an_agent_has_no_scenes(root: MemoryRoot) -> None:
    agent = Scope(app_id="app", project_id="p", owner_id="a1", owner_type="agent")
    with pytest.raises(ValueError, match="user or a group"):
        await MarkdownSceneStore(root).append(
            agent, _scene(["ac_1"]), supersedes=None, only_if=NO_SOURCE_CHECK
        )
