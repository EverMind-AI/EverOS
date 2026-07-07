"""End-to-end cascade scenarios beyond the happy-path append.

Each test boots the full cascade (writer → watchdog → md_change_state →
worker → LanceDB) against a tmp memory_root and asserts md/LanceDB
convergence after a specific perturbation. Scanner interval is held
at 60s here so the watcher path is the one being exercised — the
scanner-fallback variants live in :mod:`test_cascade_scanner_fallback`.

Coverage targets
----------------
* Rename: in-bucket / out-of-glob / cross-owner ``mv`` of a real md
  file (not the atomic-replace one — that one's covered by
  :mod:`test_cascade_fsevents_repro`).
* Content edits: re-writing an existing entry's body must flip
  ``content_sha256`` and trigger LanceDB re-upsert (not skip).
* Isolation: concurrent writes to N different owners must not bleed
  across each other's md_paths in LanceDB.
* Lap race: ``writer.append`` calls overlapping a worker's
  in-flight handler must all converge once drained, no entries lost.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from sqlmodel import SQLModel

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MarkdownReader, MarkdownWriter, MemoryRoot
from everos.infra.persistence.lancedb import (
    atomic_fact_repo,
    dispose_connection,
    ensure_business_indexes,
)
from everos.infra.persistence.lancedb.lancedb_manager import get_table
from everos.infra.persistence.lancedb.tables.atomic_fact import AtomicFact
from everos.infra.persistence.markdown import AtomicFactWriter
from everos.infra.persistence.sqlite import (
    dispose_engine,
    get_engine,
    md_change_state_repo,
)
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator


@pytest.fixture(autouse=True)
def _reset_lancedb_write_locks() -> None:
    """Drop the per-table write-lock pool between tests.

    ``LanceRepoBase`` stashes ``asyncio.Lock`` objects in a ClassVar dict
    keyed by table name; without a reset the lock outlives pytest-
    asyncio's function-scoped loop and the next test fails with "Lock
    bound to a different event loop". Mirrors the unit-test fixture in
    test_repository.py.
    """
    from everos.core.persistence.lancedb.repository import LanceRepoBase

    LanceRepoBase._reset_locks_for_tests()


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[0.0] * self.dim for _ in texts]


@pytest.fixture
async def cascade_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")

    await dispose_connection()
    await dispose_engine()

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await ensure_business_indexes()
    (tmp_path / "ome.toml").write_text("# test\n")

    yield MemoryRoot.default()

    await dispose_connection()
    await dispose_engine()


def _build_orchestrator(
    memory_root: MemoryRoot, *, scan_interval: float = 60.0
) -> CascadeOrchestrator:
    return CascadeOrchestrator(
        memory_root=memory_root,
        embedder=_StubEmbedder(),
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=scan_interval,
            worker_batch_size=20,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )


async def _wait_path_done(md_path: str, *, deadline: float = 15.0) -> None:
    """Wait until ``md_path`` lands in state AND reaches ``status='done'``.

    Bare ``_wait_drain`` returns immediately when the queue is empty,
    which is exactly the case right after a single ``append_entries``
    fires once but the watcher hasn't yet enqueued anything. This helper
    polls for the row first (i.e. watcher has noticed), then waits for
    terminal state, then re-checks after a short settle to absorb any
    last-second re-enqueue (e.g. atomic-replace echo).
    """
    async with asyncio.timeout(deadline):
        while True:
            row = await md_change_state_repo.get_by_id(md_path)
            if row is not None:
                break
            await asyncio.sleep(0.05)
        while True:
            row = await md_change_state_repo.get_by_id(md_path)
            if row is not None and row.status in ("done", "failed"):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)
        row = await md_change_state_repo.get_by_id(md_path)
        assert row is not None and row.status in ("done", "failed"), (
            f"path {md_path} flipped back to {row.status if row else 'NONE'} "
            f"after reaching done"
        )


async def _wait_paths_done(*md_paths: str, deadline: float = 15.0) -> None:
    await asyncio.gather(*[_wait_path_done(p, deadline=deadline) for p in md_paths])


async def _wait_drain(deadline: float = 15.0) -> None:
    """Wait for the *whole* queue to settle. Use only when you've already
    confirmed at least one path is in flight (via _wait_path_done first)."""
    async with asyncio.timeout(deadline):
        while True:
            summary = await md_change_state_repo.queue_summary()
            if summary.pending == 0:
                return
            await asyncio.sleep(0.05)


async def _count_lance_rows_md(md_path: str) -> int:
    table = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
    return await table.count_rows(filter=f"md_path = '{md_path}'")


async def _count_md_entries(absolute: Path) -> int:
    if not await anyio.Path(absolute).is_file():
        return 0
    parsed = await MarkdownReader.read(absolute)
    return len(parsed.entries)


def _atomic_fact_md_path(owner_id: str, bucket: _dt.date) -> str:
    return (
        f"default_app/default_project/users/{owner_id}/.atomic_facts/"
        f"atomic_fact-{bucket.isoformat()}.md"
    )


async def _seed_atomic_facts(
    writer: AtomicFactWriter,
    *,
    owner_id: str,
    bucket: _dt.date,
    n_items: int,
    text_prefix: str = "seed fact",
) -> None:
    items = [
        (
            {
                "owner_id": owner_id,
                "session_id": f"s_{j}",
                "timestamp": "2026-05-19T07:04:26+00:00",
                "parent_id": f"mc_{j}",
                "sender_ids": [owner_id],
            },
            {"Fact": f"{text_prefix} {j}"},
        )
        for j in range(n_items)
    ]
    await writer.append_entries(owner_id, items, date=bucket)


# ===== A. Rename scenarios =====


async def test_rename_same_owner_kind_in_bucket(
    cascade_runtime: MemoryRoot,
) -> None:
    """``mv atomic_fact-D1.md atomic_fact-D2.md`` inside the same owner+kind.

    Both paths match the kind glob. Expected: src lancedb rows cleared,
    dest md_path becomes the new home for the (entry_id, content) pairs.
    """
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator(memory_root)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        owner_id = "u_rename_a"
        bucket_src = _dt.date(2026, 5, 18)
        bucket_dest = _dt.date(2026, 5, 20)
        await _seed_atomic_facts(
            writer, owner_id=owner_id, bucket=bucket_src, n_items=5
        )
        src_md_path = _atomic_fact_md_path(owner_id, bucket_src)
        dest_md_path = _atomic_fact_md_path(owner_id, bucket_dest)
        src_absolute = memory_root.root / src_md_path
        dest_absolute = memory_root.root / dest_md_path

        await _wait_path_done(src_md_path)

        # Sanity: cascade has indexed the seed.
        assert await _count_lance_rows_md(src_md_path) == 5
        assert await _count_lance_rows_md(dest_md_path) == 0

        # Real rename — no tmp/atomic-replace involvement.
        await anyio.to_thread.run_sync(
            shutil.move, str(src_absolute), str(dest_absolute)
        )
        await _wait_paths_done(src_md_path, dest_md_path)

        assert await _count_lance_rows_md(src_md_path) == 0, "src not cleared"
        assert await _count_lance_rows_md(dest_md_path) == 5, "dest not reindexed"

        # md_change_state should reflect both sides finally settled.
        src_row = await md_change_state_repo.get_by_id(src_md_path)
        dest_row = await md_change_state_repo.get_by_id(dest_md_path)
        assert src_row is not None and src_row.status == "done"
        assert dest_row is not None and dest_row.status == "done"
    finally:
        await orchestrator.stop()


async def test_rename_out_of_kind_glob_degrades_to_delete(
    cascade_runtime: MemoryRoot,
) -> None:
    """``mv`` from inside the kind glob to a path outside it.

    Expected: src lancedb cleared (treated as deletion); dest path is
    silently ignored because ``match_kind`` rejects it.
    """
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator(memory_root)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        owner_id = "u_rename_oob"
        bucket = _dt.date(2026, 5, 18)
        await _seed_atomic_facts(writer, owner_id=owner_id, bucket=bucket, n_items=4)
        src_md_path = _atomic_fact_md_path(owner_id, bucket)
        src_absolute = memory_root.root / src_md_path
        # An obviously-out-of-glob target: hide it under a plain dir
        # that no kind spec registers.
        dest_absolute = memory_root.root / "out_of_scope" / "random.md"
        await anyio.Path(dest_absolute.parent).mkdir(parents=True, exist_ok=True)

        await _wait_path_done(src_md_path)
        assert await _count_lance_rows_md(src_md_path) == 4

        await anyio.to_thread.run_sync(
            shutil.move, str(src_absolute), str(dest_absolute)
        )
        # Wait for the src deletion to settle. The dest path is outside
        # the glob so it never enters md_change_state — can't wait on it.
        # Re-poll src until row reflects the rename.
        await asyncio.sleep(0.5)
        await _wait_drain()

        assert await _count_lance_rows_md(src_md_path) == 0
        # No row should appear for the out-of-glob target.
        src_row = await md_change_state_repo.get_by_id(src_md_path)
        assert src_row is not None and src_row.status == "done"
        # The dest path was never registered with any kind spec, so no
        # md_change_state row should exist for it.
        all_rows = await md_change_state_repo.queue_summary()
        # Spot check: pending should be 0; total rows present (done)
        # come only from the src side.
        assert all_rows.pending == 0
    finally:
        await orchestrator.stop()


async def test_rename_cross_owner_keeps_frontmatter_owner(
    cascade_runtime: MemoryRoot,
) -> None:
    """``mv users/u_a/.atomic_facts/X.md users/u_b/.atomic_facts/X.md``.

    Frontmatter ``user_id`` stays as ``u_a`` (rename doesn't rewrite the
    file). resolve_owner pulls owner_id from frontmatter, so dest
    LanceDB rows carry ``owner_id='u_a'`` even though md_path is under
    ``users/u_b/``. This reflects current design (frontmatter is the
    truth source) — surface it as a regression anchor.
    """
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator(memory_root)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        bucket = _dt.date(2026, 5, 18)
        owner_a = "u_a"
        owner_b = "u_b"
        await _seed_atomic_facts(writer, owner_id=owner_a, bucket=bucket, n_items=3)
        src_md_path = _atomic_fact_md_path(owner_a, bucket)
        dest_md_path = _atomic_fact_md_path(owner_b, bucket)
        src_absolute = memory_root.root / src_md_path
        dest_absolute = memory_root.root / dest_md_path
        await anyio.Path(dest_absolute.parent).mkdir(parents=True, exist_ok=True)

        await _wait_path_done(src_md_path)
        assert await _count_lance_rows_md(src_md_path) == 3

        await anyio.to_thread.run_sync(
            shutil.move, str(src_absolute), str(dest_absolute)
        )
        await _wait_paths_done(src_md_path, dest_md_path)

        assert await _count_lance_rows_md(src_md_path) == 0
        assert await _count_lance_rows_md(dest_md_path) == 3

        # Inspect a row from dest to confirm owner_id stays as u_a
        # (current design: frontmatter wins over md_path for owner_id).
        rows = await atomic_fact_repo.find_where(
            f"md_path = '{dest_md_path}'", limit=10
        )
        assert rows, "dest md_path has no rows"
        assert all(r.owner_id == owner_a for r in rows), (
            f"expected owner_id={owner_a} from frontmatter, "
            f"got {[r.owner_id for r in rows]}"
        )
    finally:
        await orchestrator.stop()


# ===== B. Write-pattern scenarios =====


async def test_modify_existing_entry_content_reindexes(
    cascade_runtime: MemoryRoot,
) -> None:
    """Rewriting an entry's body (same entry_id, new text) must flip
    content_sha256 and trigger re-upsert (not skip)."""
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator(memory_root)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        owner_id = "u_modify"
        bucket = _dt.date(2026, 5, 18)
        await _seed_atomic_facts(
            writer,
            owner_id=owner_id,
            bucket=bucket,
            n_items=3,
            text_prefix="ORIGINAL",
        )
        md_path = _atomic_fact_md_path(owner_id, bucket)
        absolute = memory_root.root / md_path
        await _wait_path_done(md_path)
        rows_before = await atomic_fact_repo.find_where(
            f"md_path = '{md_path}'", limit=10
        )
        assert len(rows_before) == 3
        sha_before = {r.entry_id: r.content_sha256 for r in rows_before}
        fact_before = {r.entry_id: r.fact for r in rows_before}

        # Read, replace body text, atomic-write back through writer.write()
        text = await anyio.Path(absolute).read_text(encoding="utf-8")
        new_text = text.replace("ORIGINAL", "EDITED")
        assert new_text != text
        mw = MarkdownWriter(memory_root)
        await mw.write(absolute, new_text)
        # The edit reuses md_path; watcher enqueue can lag behind the write,
        # so queue-empty is not a sufficient barrier. Poll for the externally
        # visible condition this scenario cares about: LanceDB rows reflect the
        # rewritten entry bodies.
        async with asyncio.timeout(15.0):
            while True:
                await _wait_drain()
                rows_after = await atomic_fact_repo.find_where(
                    f"md_path = '{md_path}'", limit=10
                )
                if len(rows_after) == 3 and all(
                    r.content_sha256 != sha_before.get(r.entry_id)
                    and "EDITED" in r.fact
                    and "ORIGINAL" not in r.fact
                    for r in rows_after
                ):
                    break
                await asyncio.sleep(0.05)

        assert len(rows_after) == 3
        sha_after = {r.entry_id: r.content_sha256 for r in rows_after}
        fact_after = {r.entry_id: r.fact for r in rows_after}

        # Every entry_id present in both, every content_sha256 changed,
        # every fact text now reflects EDITED.
        assert set(sha_after) == set(sha_before)
        for eid, sha in sha_after.items():
            assert sha != sha_before[eid], (
                f"content_sha256 did not change for {eid}: stayed {sha}"
            )
            assert "EDITED" in fact_after[eid], (
                f"fact text not updated for {eid}: {fact_after[eid]!r}"
            )
            assert "ORIGINAL" not in fact_after[eid]
            assert "ORIGINAL" in fact_before[eid]
    finally:
        await orchestrator.stop()


async def test_concurrent_writes_different_owners_no_bleed(
    cascade_runtime: MemoryRoot,
) -> None:
    """N owners writing in parallel must converge with per-md_path
    isolation: each md_path holds exactly its owner's entries."""
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator(memory_root)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        bucket = _dt.date(2026, 5, 18)
        owners = [f"u_concur_{i}" for i in range(5)]
        per_owner = 4

        await asyncio.gather(
            *[
                _seed_atomic_facts(
                    writer,
                    owner_id=oid,
                    bucket=bucket,
                    n_items=per_owner,
                    text_prefix=f"by-{oid}",
                )
                for oid in owners
            ]
        )
        md_paths = [_atomic_fact_md_path(oid, bucket) for oid in owners]
        await _wait_paths_done(*md_paths)

        for oid in owners:
            md_path = _atomic_fact_md_path(oid, bucket)
            rows = await atomic_fact_repo.find_where(f"md_path = '{md_path}'", limit=10)
            assert len(rows) == per_owner, (
                f"{oid}: expected {per_owner} rows, got {len(rows)}"
            )
            # Every row in this md_path must belong to this owner —
            # no bleed from another concurrent owner's writes.
            assert all(r.owner_id == oid for r in rows)
            assert all(f"by-{oid}" in r.fact for r in rows)
    finally:
        await orchestrator.stop()


async def test_lap_append_during_handler_no_loss(
    cascade_runtime: MemoryRoot,
) -> None:
    """Writer keeps appending while worker is mid-handler.

    Slow the embedder so a handler invocation overlaps later appends.
    On drain, lance_rows must equal md entries — the lap is absorbed
    by the worker's status='processing' guard + re-claim.
    """
    memory_root = cascade_runtime

    class _SlowEmbedder(_StubEmbedder):
        async def embed(self, text: str) -> list[float]:
            await asyncio.sleep(0.05)  # handler takes ~0.05*N entries
            return [0.0] * self.dim

    orchestrator = CascadeOrchestrator(
        memory_root=memory_root,
        embedder=_SlowEmbedder(),
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=60.0,
            worker_batch_size=20,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        owner_id = "u_lap"
        bucket = _dt.date(2026, 5, 18)
        total = 30
        for i in range(total):
            await writer.append_entries(
                owner_id,
                [
                    (
                        {
                            "owner_id": owner_id,
                            "session_id": f"s_{i}",
                            "timestamp": "2026-05-19T07:04:26+00:00",
                            "parent_id": f"mc_{i}",
                            "sender_ids": [owner_id],
                        },
                        {"Fact": f"fact body {i}"},
                    )
                ],
                date=bucket,
            )
            # Pace just slow enough that some writes land during a
            # handler invocation (~50ms per embed), but fast enough
            # that multiple writes accumulate during one handler.
            await asyncio.sleep(0.02)

        md_path = _atomic_fact_md_path(owner_id, bucket)
        absolute = memory_root.root / md_path
        await _wait_path_done(md_path, deadline=30.0)

        md_entries = await _count_md_entries(absolute)
        lance_rows = await _count_lance_rows_md(md_path)
        assert md_entries == total, (
            f"writer self-check: expected {total} md entries, got {md_entries}"
        )
        assert lance_rows == md_entries, f"LAP LOSS: md={md_entries} lance={lance_rows}"
    finally:
        await orchestrator.stop()


# ===== C. Scanner fallback scenarios =====


def _build_orchestrator_fast_scanner(memory_root: MemoryRoot) -> CascadeOrchestrator:
    """Same as :func:`_build_orchestrator` but with a 2s scanner so tests
    don't wait 30s for the fallback path."""
    return CascadeOrchestrator(
        memory_root=memory_root,
        embedder=_StubEmbedder(),
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=2.0,
            worker_batch_size=20,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )


def _silence_handler_method(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Replace ``watcher._Handler.<name>`` with a no-op for the duration
    of the test. Simulates fseventsd missing that event class entirely.
    """
    from everos.memory.cascade import watcher as watcher_module

    monkeypatch.setattr(
        watcher_module._Handler,
        name,
        lambda self, event: None,
    )


async def test_scanner_recovers_missed_delete(
    cascade_runtime: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watcher's ``on_deleted`` is silenced → unlink no longer enqueues
    via the watcher. The scanner sweep should still notice the path
    missing on disk and enqueue a 'deleted' on its own."""
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator_fast_scanner(memory_root)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        owner_id = "u_scan_del"
        bucket = _dt.date(2026, 5, 18)
        await _seed_atomic_facts(writer, owner_id=owner_id, bucket=bucket, n_items=3)
        md_path = _atomic_fact_md_path(owner_id, bucket)
        absolute = memory_root.root / md_path
        await _wait_path_done(md_path)
        assert await _count_lance_rows_md(md_path) == 3

        # From here on, watcher ignores deletions.
        _silence_handler_method(monkeypatch, "on_deleted")

        absolute.unlink()
        # Watcher won't enqueue; scanner sweeps every 2s and should
        # spot mtime/existence inconsistency, then enqueue 'deleted'.
        await asyncio.sleep(0.2)

        async def _lance_cleared() -> bool:
            return await _count_lance_rows_md(md_path) == 0

        async with asyncio.timeout(10.0):
            while not await _lance_cleared():  # noqa: ASYNC110 - polling cascade state
                await asyncio.sleep(0.1)

        async with asyncio.timeout(5.0):
            while True:
                row = await md_change_state_repo.get_by_id(md_path)
                if row is not None and row.status == "done":
                    break
                await asyncio.sleep(0.1)
        assert row.change_type == "deleted"
    finally:
        await orchestrator.stop()


async def test_scanner_indexes_preexisting_md(
    cascade_runtime: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An md file written BEFORE cascade starts (or by an editor while
    cascade is offline). watchdog ignores files that exist at schedule
    time — only the scanner can pick it up. Simulate by silencing
    on_created and writing the file before orchestrator.start()."""
    memory_root = cascade_runtime

    # Pre-seed: write the md directly to disk before any cascade is up.
    owner_id = "u_scan_pre"
    bucket = _dt.date(2026, 5, 18)
    writer = AtomicFactWriter(root=memory_root)
    await _seed_atomic_facts(writer, owner_id=owner_id, bucket=bucket, n_items=2)
    md_path = _atomic_fact_md_path(owner_id, bucket)
    assert (memory_root.root / md_path).is_file()

    # Now start cascade with the file already on disk. Belt-and-
    # suspenders: silence all watcher events so the only path to
    # discovery is the scanner.
    orchestrator = _build_orchestrator_fast_scanner(memory_root)
    for name in ("on_created", "on_modified", "on_moved", "on_deleted"):
        _silence_handler_method(monkeypatch, name)
    await orchestrator.start()

    try:

        async def _lance_filled() -> bool:
            return await _count_lance_rows_md(md_path) == 2

        async with asyncio.timeout(10.0):
            while not await _lance_filled():  # noqa: ASYNC110 - polling cascade state
                await asyncio.sleep(0.1)
    finally:
        await orchestrator.stop()


async def test_scanner_recovers_missed_modify(
    cascade_runtime: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All non-deletion watcher events silenced. writer.append produces
    an atomic-replace whose events are all dropped by the watcher.
    Scanner should still notice the new file and enqueue 'added'."""
    memory_root = cascade_runtime
    orchestrator = _build_orchestrator_fast_scanner(memory_root)

    # Silence everything BEFORE start() so the initial schedule doesn't
    # see any add/create events either.
    for name in ("on_created", "on_modified", "on_moved"):
        _silence_handler_method(monkeypatch, name)

    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        owner_id = "u_scan_mod"
        bucket = _dt.date(2026, 5, 18)
        await _seed_atomic_facts(writer, owner_id=owner_id, bucket=bucket, n_items=3)
        md_path = _atomic_fact_md_path(owner_id, bucket)

        async def _lance_filled() -> bool:
            return await _count_lance_rows_md(md_path) == 3

        async with asyncio.timeout(10.0):
            while not await _lance_filled():  # noqa: ASYNC110 - polling cascade state
                await asyncio.sleep(0.1)

        row = await md_change_state_repo.get_by_id(md_path)
        assert row is not None and row.status == "done"
    finally:
        await orchestrator.stop()
