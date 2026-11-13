"""Opening a store that a previous server left mid-way.

A buffer that was never flushed, a cascade that never ran or never finished
(markdown written, ledger rows left ``processing`` by a dead claimer), and a
store stamped by another major version. A restart must pick up what was left
-- each marker live exactly once, markdown and the index in agreement -- or,
for a store it does not own, refuse to start and change nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from everos.core.errors import StoreVersionError
from tests.fakes.storage import open_storage_view
from tests.helpers.faults import FaultServer, assert_exactly_once


async def test_a_buffer_left_unflushed_is_flushed_after_a_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    async with FaultServer.open(root) as srv:
        resp, sent = await srv.say("s1", "u1", "B")
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["status"] == "accumulated"
    async with FaultServer.open(root) as srv:
        assert srv.episodes("u1") == {}
        flushed = await srv.flush("s1")
        assert flushed.json()["data"]["status"] == "extracted", flushed.text
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await srv.consistent()


async def test_a_queued_batch_left_behind_is_delivered_after_a_restart(
    tmp_path: Path,
) -> None:
    """``async_mode``: the 202 promised the batch; a restart must keep it."""
    root = tmp_path / "root"
    async with FaultServer.open(root) as srv:
        srv.upstream.cut_on_add = True
        srv.upstream.delay["_BatchReply"] = 30.0  # the worker stays busy
        resp, sent = await srv.say("s1", "u1", "Q", async_mode=True)
        assert resp.status_code == 202, resp.text
        resp2, sent2 = await srv.say("s1", "u1", "R", async_mode=True, at=60_000)
        assert resp2.status_code == 202, resp2.text
        srv.upstream.delay.clear()
    async with FaultServer.open(root) as srv:
        srv.upstream.cut_on_add = True
        await srv.settle(45.0)
        assert (await srv.flush("s1")).status_code in {200, 202}
        await srv.settle(45.0)
        assert_exactly_once(srv.episodes("u1"), [*sent, *sent2])
        await srv.consistent()


async def test_markdown_written_while_the_cascade_was_off_is_indexed_on_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    async with FaultServer.open(root, env={"EVEROS_CASCADE__ENABLED": "false"}) as srv:
        _resp, sent = await srv.say("s1", "u1", "C")
        assert (await srv.flush("s1")).status_code == 200
        assert_exactly_once(srv.episodes("u1"), sent)
    async with FaultServer.open(root) as srv:
        await srv.settle()
        assert await srv.searchable("u1") >= set(sent)
        await srv.consistent()


async def test_ledger_rows_left_processing_are_reclaimed_on_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    async with FaultServer.open(root, env={"EVEROS_CASCADE__ENABLED": "false"}) as srv:
        _resp, sent = await srv.say("s1", "u1", "P")
        assert (await srv.flush("s1")).status_code == 200
    # A claimer that died: every episode row claimed, never finished.
    view = open_storage_view(root)
    try:
        ledger = view.sqlite.md_change_state
        for path in root.rglob("episode-*.md"):
            md_path = path.relative_to(root).as_posix()
            await ledger.upsert(md_path, kind="episode", change_type="added",
                                mtime=path.stat().st_mtime)  # fmt: skip
            claimed = await ledger.claim_one(md_path)
            assert claimed is not None and claimed.status == "processing"
    finally:
        await view.aclose()
    async with FaultServer.open(root) as srv:
        await srv.settle()
        summary = await srv.ledger()
        assert summary.pending == 0 and summary.failed_permanent == 0, summary
        assert await srv.searchable("u1") >= set(sent)
        await srv.consistent()


def _digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("stamp", ["1", "3"])
async def test_a_store_of_another_version_is_refused_and_left_alone(
    tmp_path: Path, stamp: str
) -> None:
    root = tmp_path / "root"
    async with FaultServer.open(root) as srv:
        _resp, _ = await srv.say("s1", "u1", "V")
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
    (root / ".index" / "store_version").write_text(f"{stamp}\n", encoding="utf-8")
    before = _digest(root)
    with pytest.raises(StoreVersionError):
        async with FaultServer.open(root):
            pytest.fail("the server must not start over another version's store")
    assert _digest(root) == before
