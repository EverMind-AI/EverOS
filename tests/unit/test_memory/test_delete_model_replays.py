"""Counterexamples of the memory-delete TLA+ models, replayed on the real code.

Each test pins the mechanism one ablation config removed: the model found a
trace where a deleted memory comes back (or a cluster keeps a deleted
member) without it. The test drives that trace's decisive interleaving
deterministically -- the deletion step is applied directly to the stores at
the point the counterexample puts it -- and asserts the mechanism stops it.
Removing the mechanism turns the test red.

Model sources: ``DeleteMd.tla`` / ``Fence.tla`` and their ``MD_Abl_*`` /
``Fence_*`` configs (kept with the design notes, not in this repository).
Replayed elsewhere: ``MD_Naive_BD`` and ``MD_Abl_SDClearsKeys`` in
``tests/unit/test_infra/test_markdown/test_deletion.py``;
``MD_Abl_ClosureFirst`` and ``MD_Abl_ClearKeysEarly`` in
``tests/unit/test_service/test_delete_by_ids.py``; ``Fence_Naive`` and
``Fence_SpecV2`` in ``tests/unit/test_infra/test_cascade/`` (the fence
permutations, the projection's recheck, the eraser's requeue).
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from everalgo.types import Episode as AlgoEpisode

from everos.core.locks import LOCKS, owner_key
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownEpisodeStore,
    MarkdownReader,
)
from everos.memory.events import (
    EpisodeExtracted,
    ReflectedEpisode,
    ReflectionContext,
)
from everos.memory.models import Cluster, Episode, MemCell
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, Stored
from everos.memory.reflection import ReflectionFinisher, ReflectionOrchestrator
from everos.memory.strategies import trigger_profile_clustering
from tests.fakes import make_deps, make_indexes, make_stores
from tests.fakes.stores import InMemoryMemcellStore

_TS = 1_747_216_800_000
_USER = Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")
_OWNER = owner_key("app", "proj", "user", "u1")
_GONE = "ep_20250514_0099"
"""An entry id no file holds: what a deleted episode looks like to a reader."""


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _cell() -> MemCell:
    return MemCell(items=[], timestamp=_TS)


async def _write_episode(
    store: MarkdownEpisodeStore, *, memcell_id: str, text: str = "we talked"
) -> MemoryRef:
    ref = await store.append(
        _USER,
        Episode(
            owner_id="u1",
            episode=text,
            timestamp=_TS,
            subject="Chat",
            summary="short",
            session_id="s1",
            sender_ids=["u1"],
            parent_id=memcell_id,
        ),
        only_if=NO_SOURCE_CHECK,
    )
    assert ref is not None
    return ref


def _episode_event(ref: MemoryRef, *, memcell_id: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=ref.entry_id,
        episode_text="we talked",
        episode_timestamp_ms=_TS,
        owner_id="u1",
        session_id="s1",
        app_id="app",
        project_id="proj",
    )


class _ReadThenDelete(MarkdownEpisodeStore):
    """``exists`` answers from the files, then a deletion bumps the counter.

    Puts a deletion exactly between the merge's source read and its locked
    write: the read saw every source, the deletion ran right after it.
    """

    async def exists(self, scope: Scope, entry_ids: Collection[str]) -> set[str]:
        found = await super().exists(scope, entry_ids)
        LOCKS.bump_delete_count(_OWNER)
        return found


class _DeleteBeforeWrite(MarkdownEpisodeStore):
    """A deletion bumps the counter after the source read, before the write."""

    async def append_many(self, scope: Scope, episodes: Any, *, only_if: Any) -> Any:
        LOCKS.bump_delete_count(_OWNER)
        return await super().append_many(scope, episodes, only_if=only_if)


async def _merge_once(
    root: MemoryRoot, episode_store: MarkdownEpisodeStore
) -> list[EpisodeExtracted]:
    """Run one reflection over a two-member cluster whose sources all exist."""
    writer = MarkdownEpisodeStore(root)
    memcells = InMemoryMemcellStore()
    refs = []
    for i in (1, 2):
        memcells.seed(f"mc_{i}", _cell())
        refs.append(await _write_episode(writer, memcell_id=f"mc_{i}", text=f"e{i}"))
    stores = make_stores(episode=episode_store, memcell=memcells)
    stores.cluster.seed(  # type: ignore[attr-defined]
        _USER,
        "user_memory",
        Cluster(
            id="cl_1",
            members=[r.entry_id for r in refs],
            count=2,
            centroid=(1.0,),
            last_ts=_TS,
        ),
    )
    indexes = make_indexes()
    indexes.episode.seed(
        *[
            Stored(
                ref=r,
                scope=_USER,
                item=Episode(
                    owner_id="u1",
                    episode=f"e{i}",
                    timestamp=_TS,
                    subject="Chat",
                    summary="short",
                    session_id="s1",
                    parent_id=f"mc_{i}",
                ),
            )
            for i, r in enumerate(refs, 1)
        ]
    )
    reflector = AsyncMock()
    reflector.areflect = AsyncMock(
        return_value=AlgoEpisode(
            owner_id="u1", episode="merged", summary="m", subject="s", timestamp=_TS
        )
    )
    orchestrator = ReflectionOrchestrator(
        cluster_store=stores.cluster,
        episode_store=stores.episode,
        episode_index=indexes.episode,
        memcell_store=stores.memcell,
        report_store=stores.reflection_report,
        reflector=reflector,
    )
    return await orchestrator.run(scope=_USER)


async def test_merge_writes_when_nothing_was_deleted(root: MemoryRoot) -> None:
    """Positive control for the two counter tests below."""
    assert len(await _merge_once(root, MarkdownEpisodeStore(root))) == 1


async def test_merge_records_counter_before_checking_sources(
    root: MemoryRoot,
) -> None:
    """MD_Abl_CounterAfterRead / spec 1 R37: counter first, then the read."""
    assert await _merge_once(root, _ReadThenDelete(root)) == []


async def test_merge_abandoned_when_counter_moved(root: MemoryRoot) -> None:
    """MD_Abl_NoCounter / spec 2 R10: the locked write compares the counter."""
    assert await _merge_once(root, _DeleteBeforeWrite(root)) == []


async def _finish(
    root: MemoryRoot, *, keep_merged: bool, keep_sources: tuple[bool, bool]
) -> dict[str, Any]:
    """Run the reflection finish over sources e1, e2 merged into M.

    A source or M that was deleted is simply not in the file.
    """
    store = MarkdownEpisodeStore(root)
    first = await _write_episode(store, memcell_id="mc_1")
    srcs = [
        first if keep_sources[0] else replace(first, entry_id=_GONE),
        await _write_episode(store, memcell_id="mc_2")
        if keep_sources[1]
        else replace(first, entry_id=_GONE),
    ]
    merged_id = (
        (await _write_episode(store, memcell_id="cl_1", text="merged")).entry_id
        if keep_merged
        else "ep_20250514_0098"
    )
    stores = make_stores(episode=store)
    stores.cluster.seed(  # type: ignore[attr-defined]
        _USER,
        "user_memory",
        Cluster(
            id="cl_1",
            members=[s.entry_id for s in srcs],
            count=2,
            centroid=(1.0,),
            last_ts=_TS,
        ),
    )
    finisher = ReflectionFinisher(
        cluster_store=stores.cluster,
        episode_store=stores.episode,
        report_store=stores.reflection_report,
        embedder=make_deps().embedding,  # type: ignore[arg-type]
    )
    await finisher.finish(
        scope=_USER,
        merged_entry_id=merged_id,
        context=ReflectionContext(
            cluster_id="cl_1",
            mode="init",
            source_members=[(s.entry_id, "episode") for s in srcs],
            episodes=[
                ReflectedEpisode(
                    entry_id=s.entry_id, md_path=s.md_path, parent_id=f"mc_{i}"
                )
                for i, s in enumerate(srcs, 1)
            ],
            merged_episode="merged",
            last_ts=_TS,
        ),
    )
    fm = (await MarkdownReader.read(root.root / first.md_path)).frontmatter
    return dict(fm.get("deprecated_entries") or {})


async def test_finish_marks_nothing_when_merged_gone(root: MemoryRoot) -> None:
    """MD_Abl_NoFinishOnlyIf / spec 2 R6, R9: no mark points at a deleted M."""
    assert await _finish(root, keep_merged=False, keep_sources=(True, True)) == {}


async def test_finish_marks_only_sources_still_in_md(root: MemoryRoot) -> None:
    """Spec 1 R22: a source deleted before the finish gets no mark."""
    marks = await _finish(root, keep_merged=True, keep_sources=(True, False))
    assert len(marks) == 1
    assert _GONE not in marks
    assert all(entry_id.startswith("ep_") for entry_id in marks)


async def test_clustering_skips_episode_deleted_before_lock(root: MemoryRoot) -> None:
    """MD_Abl_NoClusterCheck / spec 2 R4.

    The single-id deletion removed the episode and then cleared it from its
    cluster under the clustering lock; a clustering run that queued behind
    it must not add the episode back.
    """
    store = MarkdownEpisodeStore(root)
    memcells = InMemoryMemcellStore()
    memcells.seed("mc_1", _cell())
    ref = await _write_episode(store, memcell_id="mc_1")
    gone = replace(ref, entry_id=_GONE)
    deps = make_deps(stores=make_stores(episode=store, memcell=memcells))

    events = await trigger_profile_clustering(
        _episode_event(gone, memcell_id="mc_1"), deps
    )

    assert events == []
    assert await deps.stores.cluster.list_for_owner(_USER, "user_memory") == []


async def test_write_in_flight_at_purge_is_caught_by_the_barrier(
    root: MemoryRoot, storage: Any
) -> None:
    """MD_Naive_SD / spec 1 R2: an /add episode write passed its source check
    and holds the owner lock when the session deletion purges the memcell and
    scans. The barrier makes the scan wait for that write, so it is found
    and removed; without it the scan sees nothing and the write lands after.
    """
    from everos.infra.persistence.factories import build_stores
    from everos.memory.models import MemcellRecord
    from everos.memory.ports import SessionKey

    await storage.create_schema()
    stores = build_stores(storage.root, storage.sqlite)
    key = SessionKey(app_id="app", project_id="proj", session_id="s1")
    await stores.memcell.insert_many(
        key,
        [
            MemcellRecord(
                memcell_id="mc_1",
                cell=_cell(),
                raw_type="conversation",
                message_ids=[],
                sender_ids=["u1"],
            )
        ],
    )
    checked, release = asyncio.Event(), asyncio.Event()

    async def memcell_alive_then_slow_disk() -> bool:
        alive = bool(await stores.memcell.find_by_ids(["mc_1"]))
        checked.set()
        await release.wait()  # the write itself is slow
        return alive

    write = asyncio.create_task(
        stores.episode.append(
            _USER,
            Episode(
                owner_id="u1",
                episode="late",
                timestamp=_TS,
                subject="s",
                summary="s",
                session_id="s1",
                parent_id="mc_1",
            ),
            only_if=memcell_alive_then_slow_disk,
        )
    )
    await checked.wait()
    await stores.session_purge.purge(key)  # the deletion's first step
    deletion = asyncio.create_task(stores.episode.remove_session(_USER, "s1"))
    await asyncio.sleep(0.05)  # the deletion reaches its scan (or the barrier)
    release.set()
    await write
    await deletion

    left = [
        entry.id
        for f in (root.root / "app").rglob("episodes/*.md")
        for entry in (await MarkdownReader.read(f)).entries
    ]
    assert left == [], "the in-flight write landed after the deletion's scan"
