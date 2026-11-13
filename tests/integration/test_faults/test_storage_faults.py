"""Storage that refuses a write: a full disk, a read-only root, an index write.

Each fault is raised where the real one surfaces -- the markdown writer's
staging write (``OSError`` ENOSPC / EROFS, half the file already written),
the SQLite driver (``database or disk is full``), the LanceDB write inside the
table's lock -- and each test checks what the user would: the request that hit
it is answered with an error, no staging file stays behind, the server keeps
answering, and once the storage takes writes again the memory is whole, once,
with markdown and the index in agreement.
"""

from __future__ import annotations

import asyncio
import errno
from pathlib import Path

import pytest

from tests.helpers.faults import (
    FaultServer,
    assert_exactly_once,
    index_write_fails,
    markdown_write_fails,
    sqlite_write_fails,
    wait_until,
)

# ── markdown ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EROFS])
async def test_a_failed_episode_write_is_reported_and_leaves_no_partial_file(
    tmp_path: Path, error: int
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        resp, sent = await srv.say("s1", "u1", "W")
        assert resp.status_code == 200, resp.text
        with markdown_write_fails(error, where="episode") as failed:
            flushed = await srv.flush("s1")
        assert failed, "the fault never fired"
        assert flushed.status_code >= 500, flushed.text
        assert srv.staging() == []
        assert srv.episodes("u1") == {}
        # The server keeps serving, and the documented recovery works: the
        # batch is sent again once the disk takes writes.
        assert (await srv.health())["status"] == "ok"
        resp, _ = await srv.say("s1", "u1", "W")
        assert resp.status_code == 200, resp.text
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await srv.consistent()


async def test_a_full_disk_on_the_profile_write_keeps_the_profile_whole(
    tmp_path: Path,
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        resp, first = await srv.say("s1", "u1", "P", 2, fact=True)
        assert resp.status_code == 200, resp.text
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        before = srv.profile_text("u1")
        assert all(f"Holds {m}" in before for m in first), before
        with markdown_write_fails(errno.ENOSPC, where="user.md") as failed:
            _resp, _ = await srv.say("s2", "u1", "Q", 2, fact=True, at=60_000)
            assert (await srv.flush("s2")).status_code == 200
            await srv.settle()
        assert failed, "the fault never fired"
        assert srv.profile_text("u1") == before
        assert srv.staging() == []
        await srv.consistent()


# ── SQLite ──────────────────────────────────────────────────────────────────


async def test_a_full_disk_on_the_buffer_write_refuses_the_add(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        with sqlite_write_fails(srv.application, tables=["unprocessed_buffer"]) as hit:
            resp, sent = await srv.say("s1", "u1", "S")
        assert hit, "the fault never fired"
        assert resp.status_code >= 500, resp.text
        assert (await srv.health())["status"] == "ok"
        resp, _ = await srv.say("s1", "u1", "S")
        assert resp.status_code == 200, resp.text
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await srv.consistent()


async def test_a_full_disk_on_the_memcell_write_loses_nothing(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        resp, sent = await srv.say("s1", "u1", "S")
        assert resp.status_code == 200, resp.text
        with sqlite_write_fails(srv.application, tables=["memcell"]) as hit:
            flushed = await srv.flush("s1")
        assert hit, "the fault never fired"
        assert flushed.status_code >= 500, flushed.text
        assert srv.episodes("u1") == {}
        # Nothing was cut: the buffer still holds the batch.
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), sent)
        await srv.consistent()
        assert srv.staging() == []


async def test_a_full_disk_on_the_cascade_ledger_is_caught_up(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        _resp, sent = await srv.say("s1", "u1", "L")
        with sqlite_write_fails(srv.application, tables=["md_change_state"]) as hit:
            assert (await srv.flush("s1")).status_code == 200
            await asyncio.sleep(1.5)
        assert hit, "the fault never fired"
        assert_exactly_once(srv.episodes("u1"), sent)

        # The next sweep (every second here) finds the files the watcher
        # could not record.
        async def indexed() -> bool:
            return await srv.searchable("u1") >= set(sent)

        await wait_until(indexed, 15.0)
        await srv.settle()
        await srv.consistent()


async def test_a_read_only_root_refuses_writes_and_serves_reads(tmp_path: Path) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        resp, kept = await srv.say("s0", "u1", "K")
        assert (await srv.flush("s0")).status_code == 200
        await srv.settle()
        message = "attempt to write a readonly database"
        with (
            markdown_write_fails(errno.EROFS),
            sqlite_write_fails(srv.application, message=message),
        ):
            resp, sent = await srv.say("s1", "u1", "R")
            assert resp.status_code >= 500, resp.text
            searched = await srv.search("notes", user_id="u1")
            assert searched.status_code == 200, searched.text
            assert (await srv.client.get("/health")).status_code == 200
        assert srv.staging() == []
        resp, _ = await srv.say("s1", "u1", "R")
        assert resp.status_code == 200, resp.text
        assert (await srv.flush("s1")).status_code == 200
        await srv.settle()
        assert_exactly_once(srv.episodes("u1"), [*kept, *sent])
        await srv.consistent()


# ── LanceDB ─────────────────────────────────────────────────────────────────


FULL_INDEX_DISK = RuntimeError(
    "LanceError(IO): Generic LocalFileSystem error: Unable to write: "
    "No space left on device (os error 28)"
)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "by design an index write that fails for a full disk is unrecoverable "
        "(infra/persistence/lancedb/_base/repository.py, "
        "_TRANSIENT_EXECUTION_MARKERS): the row stays failed_permanent after the "
        "disk is freed, so the memory stays unsearchable until `cascade fix` or "
        "the file changes again, while /health still says healthy"
    ),
)
@pytest.mark.parametrize(
    "error",
    [FULL_INDEX_DISK, OSError(errno.ENOSPC, "No space left on device")],
    ids=["lance-runtime-error", "oserror"],
)
async def test_an_index_write_failure_is_reported_and_retried(
    tmp_path: Path, error: BaseException
) -> None:
    async with FaultServer.open(tmp_path / "root") as srv:
        _resp, sent = await srv.say("s1", "u1", "X")
        with index_write_fails(error) as failed:
            assert (await srv.flush("s1")).status_code == 200

            async def reported() -> bool:
                cascade = (await srv.health())["cascade"]
                return bool(failed) and (
                    cascade["failed_retryable"] + cascade["failed_permanent"] > 0
                )

            await wait_until(reported)
        assert_exactly_once(srv.episodes("u1"), sent)
        await srv.settle()
        assert await srv.searchable("u1") >= set(sent)
        await srv.consistent()
