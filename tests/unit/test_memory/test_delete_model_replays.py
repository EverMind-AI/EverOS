"""Counterexamples of the memory-delete TLA+ models, replayed on the real code.

Each test pins the mechanism one ablation config removed: the model found a
trace where a deleted memory comes back without it. The test drives that
trace's decisive interleaving deterministically -- the deletion step is
applied directly to the stores at the point the counterexample puts it -- and
asserts the mechanism stops it. Removing the mechanism turns the test red.

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
from pathlib import Path
from typing import Any

import pytest

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownReader
from everos.memory.models import Episode, MemCell
from everos.memory.ports import Scope

_TS = 1_747_216_800_000
_USER = Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _cell() -> MemCell:
    return MemCell(items=[], timestamp=_TS)


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
