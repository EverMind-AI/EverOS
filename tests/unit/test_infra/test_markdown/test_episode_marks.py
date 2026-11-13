"""The episode store's marks and reads by id (2.0 removes no entry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownEpisodeStore,
    MarkdownReader,
)
from everos.memory.models import Episode
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope

_TS = 1_747_216_800_000


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _scope(project: str = "p1", owner: str = "u1") -> Scope:
    return Scope(app_id="app", project_id=project, owner_id=owner, owner_type="user")


def _episode(session_id: str | None, text: str = "talk") -> Episode:
    return Episode(
        owner_id="u1",
        episode=text,
        timestamp=_TS,
        subject="s",
        summary="s",
        session_id=session_id,
        parent_id="mc_1",
    )


async def _write(
    store: MarkdownEpisodeStore, scope: Scope, *sessions: str
) -> list[MemoryRef]:
    return await store.append_many(
        scope, [_episode(s, f"talk {s}") for s in sessions], only_if=NO_SOURCE_CHECK
    )


async def _ids(root: MemoryRoot, md_path: str) -> list[str]:
    return [e.id for e in (await MarkdownReader.read(root.root / md_path)).entries]


async def test_clear_deprecations_drops_only_the_keys(root: MemoryRoot) -> None:
    """B6 (store side): clearing drops only the named marks; the rest stay."""
    store = MarkdownEpisodeStore(root)
    a, b, c = await _write(store, _scope(), "s1", "s2", "s3")
    assert a.md_path == b.md_path == c.md_path
    for src, merged in ((a, "ep_m1"), (b, "ep_m1"), (c, "ep_m2")):
        await store.deprecate(
            _scope(),
            entries=[(src.md_path, src.entry_id)],
            replaced_by=merged,
            only_if=NO_SOURCE_CHECK,
        )

    assert await store.deprecations(_scope(), a.md_path) == {
        a.entry_id: "ep_m1",
        b.entry_id: "ep_m1",
        c.entry_id: "ep_m2",
    }
    await store.clear_deprecations(_scope(), [a.entry_id])
    assert await store.deprecations(_scope(), a.md_path) == {
        b.entry_id: "ep_m1",
        c.entry_id: "ep_m2",
    }


async def test_read_many_returns_the_episodes_in_md(root: MemoryRoot) -> None:
    store = MarkdownEpisodeStore(root)
    (a,) = await _write(store, _scope(), "s1")

    (found,) = await store.read_many(_scope(), [a.entry_id, "ep_20260101_9", "ep_x"])

    assert found.ref == a
    assert (found.item.episode, found.item.session_id) == ("talk s1", "s1")
    assert found.item.timestamp == _TS
