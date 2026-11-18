"""Removing entries from the daily logs: by session and by id."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from everos.core.locks import LOCKS, owner_key
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


async def test_remove_session_only_exact_match(root: MemoryRoot) -> None:
    """RF1: ``s1`` does not take ``s10`` (or a regex-looking id) with it."""
    store = MarkdownEpisodeStore(root)
    refs = await _write(store, _scope(), "s1", "s10", "s1", "a.b*", "s10")

    removed = await store.remove_session(_scope(), "s1")

    assert [r.entry_id for r in removed] == [refs[0].entry_id, refs[2].entry_id]
    assert await _ids(root, refs[0].md_path) == [
        refs[1].entry_id,
        refs[3].entry_id,
        refs[4].entry_id,
    ]
    assert await store.remove_session(_scope(), "a.b") == []


async def test_remove_session_keeps_other_project(root: MemoryRoot) -> None:
    """RF2: the same session id in another project is another conversation."""
    store = MarkdownEpisodeStore(root)
    await _write(store, _scope("p1"), "s1")
    (other,) = await _write(store, _scope("p2"), "s1")

    await store.remove_session(_scope("p1"), "s1")

    assert await _ids(root, other.md_path) == [other.entry_id]


async def test_remove_session_keeps_unparsable_entries(root: MemoryRoot) -> None:
    """RF5: hand-edited entries the deletion cannot read are left alone, and
    so is the text around them; nothing raises."""
    store = MarkdownEpisodeStore(root)
    refs = await _write(store, _scope(), "s1", "s1", "s1")
    path = root.root / refs[0].md_path
    text = path.read_text(encoding="utf-8")
    # entry 2 lost its session line's colon; entry 3's close marker is gone
    text = text.replace("**session_id**: s1", "**session_id** s1", 2).replace(
        "**session_id** s1", "**session_id**: s1", 1
    )
    text = text.replace(f"<!-- /entry:{refs[2].entry_id} -->", "", 1)
    path.write_text(text + "\nhand-written note\n", encoding="utf-8")

    removed = await store.remove_session(_scope(), "s1")

    assert [r.entry_id for r in removed] == [refs[0].entry_id]
    after = path.read_text(encoding="utf-8")
    assert f"<!-- entry:{refs[1].entry_id} -->" in after
    assert f"<!-- entry:{refs[2].entry_id} -->" in after
    assert "hand-written note" in after


async def test_emptied_file_kept_and_next_id_not_reused(root: MemoryRoot) -> None:
    """A12 / R25: the file outlives its last entry and the counter goes on."""
    store = MarkdownEpisodeStore(root)
    refs = await _write(store, _scope(), "s1", "s1", "s1")

    await store.remove_session(_scope(), "s1")
    assert await _ids(root, refs[0].md_path) == []
    (new,) = await _write(store, _scope(), "s2")

    assert new.entry_id not in {r.entry_id for r in refs}
    assert new.entry_id.endswith("04")


async def test_remove_session_leaves_deprecated_entries(root: MemoryRoot) -> None:
    """TLA+ MD_Abl_SDClearsKeys: a session deletion must not clear the merge
    marks of its entries -- a concurrent by-id deletion of the merged episode
    finds the merge's sources through them."""
    store = MarkdownEpisodeStore(root)
    (src, _) = await _write(store, _scope(), "s1", "s2")
    await store.deprecate(
        _scope(),
        entries=[(src.md_path, src.entry_id)],
        replaced_by="ep_merged",
        only_if=NO_SOURCE_CHECK,
    )

    await store.remove_session(_scope(), "s1")

    assert await store.merged_into(_scope(), ["ep_merged"]) == {src.entry_id}


async def test_remove_takes_the_owner_lock_and_bumps_the_counter(
    root: MemoryRoot,
) -> None:
    store = MarkdownEpisodeStore(root)
    (ref,) = await _write(store, _scope(), "s1")
    key = owner_key("app", "p1", "user", "u1")

    async with LOCKS.hold("owner", key):
        pending = asyncio.create_task(store.remove(_scope(), [ref.entry_id]))
        await asyncio.sleep(0.05)
        assert not pending.done()
    assert [r.entry_id for r in await pending] == [ref.entry_id]
    assert LOCKS.delete_count(key) == 1
    assert await store.remove(_scope(), [ref.entry_id]) == []
    assert LOCKS.delete_count(key) == 1  # nothing removed, nothing counted


async def test_concurrent_append_survives_rewrite(root: MemoryRoot) -> None:
    """A13 / TLA+ MD_Naive_BD: a rewrite that interleaves with appends of
    another session never drops them. Two store instances, as two writers of
    one owner would be: only the owner lock orders them."""
    writer = MarkdownEpisodeStore(root)
    deleter = MarkdownEpisodeStore(root)
    kept: list[MemoryRef] = []
    for _ in range(100):
        await _write(writer, _scope(), "s_del")

        async def append() -> None:
            kept.extend(await _write(writer, _scope(), "s_keep"))

        await asyncio.gather(deleter.remove_session(_scope(), "s_del"), append())

    on_disk = set(await _ids(root, kept[0].md_path))
    assert {r.entry_id for r in kept} <= on_disk
    assert len(on_disk) == len(kept)


async def test_merged_into_and_clear_deprecations(root: MemoryRoot) -> None:
    """B6 (store side): one step of the closure; clearing drops only the keys."""
    store = MarkdownEpisodeStore(root)
    a, b, c = await _write(store, _scope(), "s1", "s2", "s3")
    for src, merged in ((a, "ep_m1"), (b, "ep_m1"), (c, "ep_m2")):
        await store.deprecate(
            _scope(),
            entries=[(src.md_path, src.entry_id)],
            replaced_by=merged,
            only_if=NO_SOURCE_CHECK,
        )

    assert await store.merged_into(_scope(), ["ep_m1"]) == {a.entry_id, b.entry_id}
    await store.clear_deprecations(_scope(), [a.entry_id])
    assert await store.merged_into(_scope(), ["ep_m1", "ep_m2"]) == {
        b.entry_id,
        c.entry_id,
    }


async def test_read_many_returns_the_episodes_in_md(root: MemoryRoot) -> None:
    store = MarkdownEpisodeStore(root)
    a, b = await _write(store, _scope(), "s1", "s2")
    await store.remove(_scope(), [b.entry_id])

    (found,) = await store.read_many(_scope(), [a.entry_id, b.entry_id, "ep_x"])

    assert found.ref == a
    assert (found.item.episode, found.item.session_id) == ("talk s1", "s1")
    assert found.item.timestamp == _TS
