"""``Deleter.delete_by_ids`` over real markdown and SQLite stores."""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from typing import Any

import pytest
import structlog.testing

from everos.config import MemorizeSettings
from everos.core.errors import DeletionIncompleteError
from everos.infra.persistence.factories import build_stores
from everos.infra.persistence.markdown import MarkdownEpisodeStore
from everos.memory.deps import Stores
from everos.memory.models import Cluster, Episode
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope
from everos.service.memory._delete import Deleter
from tests.fakes.storage import Storage

_U1 = Scope(app_id="a", project_id="p", owner_id="u1", owner_type="user")
_TS = 1_747_216_800_000


class _Eraser:
    def __init__(self) -> None:
        self.erased: list[dict[str, list[str]]] = []

    async def erase(
        self, scope: Scope, removed: Mapping[str, Sequence[MemoryRef]]
    ) -> None:
        self.erased.append({k: [r.entry_id for r in v] for k, v in removed.items()})

    async def erase_owner(self, scope: Scope, kinds: Any) -> None:
        raise AssertionError("not a whole-owner deletion")


@pytest.fixture
async def stores(storage: Storage) -> AsyncIterator[Stores]:
    await storage.create_schema()
    yield build_stores(storage.root, storage.sqlite)


def _deleter(stores: Stores, eraser: _Eraser | None = None) -> Deleter:
    return Deleter(
        stores=stores,
        eraser=eraser or _Eraser(),  # type: ignore[arg-type]
        settings=MemorizeSettings(),
    )


async def _episodes(
    stores: Stores, scope: Scope, *texts: str, ts: int = _TS
) -> list[MemoryRef]:
    return await stores.episode.append_many(
        scope,
        [
            Episode(
                owner_id=scope.owner_id,
                episode=t,
                timestamp=ts + i * 1_000,
                subject="s",
                summary="s",
                session_id="s1",
                parent_id=f"mc_{t}",
            )
            for i, t in enumerate(texts)
        ],
        only_if=NO_SOURCE_CHECK,
    )


async def _merge(
    stores: Stores, scope: Scope, sources: list[MemoryRef], into: MemoryRef
) -> None:
    await stores.episode.deprecate(
        scope,
        entries=[(s.md_path, s.entry_id) for s in sources],
        replaced_by=into.entry_id,
        only_if=NO_SOURCE_CHECK,
    )


async def _by_ids(
    d: Deleter, *ids: str, project: str = "p", note: str | None = None
) -> None:
    await d.delete_by_ids(
        app_id="a",
        project_id=project,
        memory_type="episode",
        ids=list(ids),
        reason_code="outdated",
        note=note,
    )


async def _left(stores: Stores, scope: Scope, refs: Sequence[MemoryRef]) -> set[str]:
    return await stores.episode.exists(scope, [r.entry_id for r in refs])


async def test_deletes_the_episode_and_its_merge_sources(stores: Stores) -> None:
    """B3 / B6: deleting M takes what was merged into it."""
    e1, e2, keep = await _episodes(stores, _U1, "e1", "e2", "keep")
    (m,) = await _episodes(stores, _U1, "merged")
    await _merge(stores, _U1, [e1, e2], m)
    eraser = _Eraser()

    await _by_ids(_deleter(stores, eraser), m.id)

    assert await _left(stores, _U1, [e1, e2, keep, m]) == {keep.entry_id}
    assert await stores.episode.merged_into(_U1, [m.entry_id]) == set()  # keys cleared
    (erased,) = eraser.erased
    assert set(erased) == {"episode"}
    assert sorted(erased["episode"]) == sorted([m.entry_id, e1.entry_id, e2.entry_id])


class _MarkDuringRemove(MarkdownEpisodeStore):
    """A Reflection finish marks a source into M while step 1 removes M."""

    source: MemoryRef
    merged: MemoryRef
    armed = True

    async def remove(self, scope: Scope, entry_ids: Collection[str]) -> list[MemoryRef]:
        if self.armed and self.merged.entry_id in entry_ids:
            self.armed = False
            await self.deprecate(
                scope,
                entries=[(self.source.md_path, self.source.entry_id)],
                replaced_by=self.merged.entry_id,
                only_if=NO_SOURCE_CHECK,
            )
        return await super().remove(scope, entry_ids)


async def test_closure_includes_sources_marked_after_r_removed(
    storage: Storage, stores: Stores
) -> None:
    """TLA+ MD_Abl_ClosureFirst / spec 2 R9: the closure is read after the
    requested episode is gone, so a mark that landed just before is seen."""
    store = _MarkDuringRemove(storage.root)
    late, keep = await _episodes(stores, _U1, "late", "keep")
    (m,) = await _episodes(stores, _U1, "merged")
    store.source, store.merged = late, m
    stores = build_stores(storage.root, storage.sqlite)
    object.__setattr__(stores, "episode", store)

    await _by_ids(_deleter(stores), m.id)

    assert await _left(stores, _U1, [late, keep, m]) == {keep.entry_id}


class _CrashAfterStep1(MarkdownEpisodeStore):
    crashed = False

    async def merged_into(self, scope: Scope, targets: Collection[str]) -> set[str]:
        if not self.crashed:
            self.crashed = True
            raise OSError("process died")
        return await super().merged_into(scope, targets)


async def test_resend_after_step1_still_finds_sources(
    storage: Storage, stores: Stores
) -> None:
    """TLA+ MD_Abl_ClearKeysEarly / spec 2 R15: after a crash right after M
    went, the resend finds M's sources through the marks kept until step 4."""
    e1, e2 = await _episodes(stores, _U1, "e1", "e2")
    (m,) = await _episodes(stores, _U1, "merged")
    await _merge(stores, _U1, [e1, e2], m)
    store = _CrashAfterStep1(storage.root)
    stores = build_stores(storage.root, storage.sqlite)
    object.__setattr__(stores, "episode", store)
    d = _deleter(stores)

    with pytest.raises(DeletionIncompleteError):
        await _by_ids(d, m.id)
    assert await _left(stores, _U1, [m]) == set()
    await _by_ids(d, m.id)

    assert await _left(stores, _U1, [e1, e2]) == set()


async def _cluster(stores: Stores, members: list[MemoryRef], cid: str = "cl_1") -> None:
    await stores.cluster.upsert(
        _U1,
        "user_memory",
        Cluster(
            id=cid,
            centroid=(0.5, 0.5),
            count=len(members),
            last_ts=1,
            preview=["old"],
            members=[m.entry_id for m in members],
        ),
        member_type="episode",
    )


async def test_detach_fields_match_remaining_members(stores: Stores) -> None:
    """B5 / C13: count, last_ts and preview (newest five, full text) follow
    the members left; the centroid stays."""
    refs = await _episodes(stores, _U1, *[f"text {i}" for i in range(8)])
    await _cluster(stores, refs)

    await _by_ids(_deleter(stores), refs[7].id, refs[0].id)

    cluster = await stores.cluster.get("cl_1")
    assert cluster is not None
    assert sorted(cluster.members) == sorted(r.entry_id for r in refs[1:7])
    assert cluster.count == 6
    assert cluster.last_ts == _TS + 6 * 1_000
    assert cluster.preview == [f"text {i}" for i in range(2, 7)]
    assert cluster.centroid == (0.5, 0.5)


async def test_dangling_member_detached_and_empty_cluster_deleted(
    stores: Stores,
) -> None:
    """R21: a member already gone from md is dropped too; with nothing left
    the cluster itself goes (no max() of an empty list)."""
    a, b = await _episodes(stores, _U1, "a", "b")
    await _cluster(stores, [a, b])
    await stores.episode.remove(_U1, [b.entry_id])  # dangling

    await _by_ids(_deleter(stores), a.id)

    assert await stores.cluster.get("cl_1") is None


async def test_unknown_ids_skipped_and_logged_without_note(stores: Stores) -> None:
    """B2 / B12: ids that match nothing are skipped; the log names hits and
    skips and never carries the note or any text."""
    (real,) = await _episodes(stores, _U1, "secret text")
    with structlog.testing.capture_logs() as logs:
        await _by_ids(
            _deleter(stores),
            real.id,
            "nobody_ep_20250101_0001",
            "u1_ep_19990101_0001",
            "u1_ep_20250101_0099",
            real.id,
            note="my private reason",
        )

    (line,) = [e for e in logs if e["event"] == "memory_delete_by_ids"]
    assert line["hit"] == [real.id]
    assert line["skipped"] == [
        "nobody_ep_20250101_0001",
        "u1_ep_19990101_0001",
        "u1_ep_20250101_0099",
    ]
    assert line["reason"] == "outdated"
    assert "my private reason" not in str(logs) and "secret text" not in str(logs)


async def test_other_project_same_entry_id_untouched(stores: Stores) -> None:
    """TB2.1: the same owner and entry id in another project is another episode."""
    (here,) = await _episodes(stores, _U1, "here")
    other_scope = Scope(app_id="a", project_id="q", owner_id="u1", owner_type="user")
    (there,) = await _episodes(stores, other_scope, "there")
    assert here.id == there.id

    await _by_ids(_deleter(stores), here.id)

    assert await _left(stores, other_scope, [there]) == {there.entry_id}
