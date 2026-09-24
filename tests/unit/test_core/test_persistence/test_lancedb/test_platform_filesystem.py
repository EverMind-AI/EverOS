"""LanceDB against filesystem semantics that differ by platform.

Every test here passes trivially on POSIX and exists to be run by the Windows
CI job, where NTFS and Win32 change the rules a local store leans on: a file
with an open handle cannot be deleted or renamed, ``rmdir`` on a directory
something else holds is a sharing violation, and a path past 260 characters
needs an opt-in. None of these are Lance bugs. They are the reasons a store
that works on a Mac can fail the first time a Windows user deletes, moves or
deeply nests their memory root.
"""

from __future__ import annotations

import datetime as dt
import gc
import inspect
import os
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

from everos.config import LanceDBSettings
from everos.core.persistence import (
    BaseLanceTable,
    MemoryRoot,
    Vector,
    open_lancedb_connection,
)
from everos.core.persistence.lancedb import LanceDailyLogRepoBase, LanceRepoBase
from everos.core.persistence.lancedb.repository import (
    _HUSK_MIN_AGE_SECONDS,
    _remove_empty_index_dirs,
)


class _Probe(BaseLanceTable):
    TABLE_NAME: ClassVar[str] = "_probe"

    id: str
    owner_id: str
    app_id: str = "default"
    project_id: str = "default"
    entry_id: str
    session_id: str = "s"
    parent_type: str = "memcell"
    parent_id: str = "mc"
    md_path: str = "users/u/notes/x.md"
    text: str = "x"
    vector: Vector(4)  # type: ignore[valid-type]


class _ProbeRepo(LanceDailyLogRepoBase[_Probe]):
    schema = _Probe


def _rows(n: int, *, prefix: str = "e") -> list[_Probe]:
    return [
        _Probe(
            id=f"u_{prefix}{i}",
            owner_id="u",
            entry_id=f"{prefix}{i}",
            vector=[1.0, 0, 0, 0],
        )
        for i in range(n)
    ]


async def _close(obj: Any) -> None:
    """``AsyncTable.close`` / ``AsyncConnection.close`` are sync in 0.34; stay
    correct if a later release makes them awaitable."""
    r = obj.close()
    if inspect.isawaitable(r):
        await r


@pytest.fixture(autouse=True)
def _reset_write_locks() -> None:
    LanceRepoBase._reset_locks_for_tests()


# ── sharing violations ──────────────────────────────────────────────────────


def test_husk_sweep_skips_a_refused_rmdir_and_keeps_going(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows an ``rmdir`` of a directory another process holds open is
    ``PermissionError``, not ``ENOTEMPTY``. One refused husk must not abort
    the sweep of the others -- the sweep is best-effort by contract."""
    indices = tmp_path / "_indices"
    husks = [indices / n for n in ("aaa", "bbb", "ccc")]
    for h in husks:
        h.mkdir(parents=True)
        old = time.time() - _HUSK_MIN_AGE_SECONDS * 2
        os.utime(h, (old, old))

    real_rmdir = Path.rmdir

    def refusing_rmdir(self: Path) -> None:
        if self.name == "bbb":
            raise PermissionError(32, "The process cannot access the file", str(self))
        real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", refusing_rmdir)

    removed = _remove_empty_index_dirs(
        str(tmp_path), live_uuids=frozenset(), min_age_seconds=1.0
    )

    assert removed == 2
    assert not husks[0].exists() and not husks[2].exists()
    assert husks[1].exists(), "the refused one is left for the next sweep"


# ── handle release ──────────────────────────────────────────────────────────


async def test_table_files_are_releasable_after_close(tmp_path: Path) -> None:
    """Deleting or moving the memory root is an ordinary user action.

    POSIX lets you unlink an open file; Windows does not. If Lance keeps a
    handle or mapping alive past ``close()``, ``rmtree`` fails with
    ``PermissionError`` on the first Windows user who tries -- and pytest's
    own tmp_path cleanup would have hidden it, since that runs at the *next*
    session start.
    """
    mr = MemoryRoot(tmp_path / "root")
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table(_Probe.TABLE_NAME, schema=_Probe)
    await table.add([r.model_dump() for r in _rows(2)])
    assert await table.count_rows() == 2

    await _close(table)
    await _close(conn)
    del table, conn
    gc.collect()

    shutil.rmtree(mr.lancedb_dir)
    assert not mr.lancedb_dir.exists()


async def test_prune_completes_while_another_handle_reads_the_table(
    tmp_path: Path,
) -> None:
    """A second connection holds the table open while prune reclaims old
    versions. On Windows the reclaimed files may be the ones the reader has
    mapped; the store must still finish and both handles must agree on the
    rows afterwards."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    writer_conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await writer_conn.create_table(_Probe.TABLE_NAME, schema=_Probe)
    repo = _ProbeRepo(table=table)
    await repo.add(_rows(3, prefix="a"))
    await repo.add(_rows(3, prefix="b"))  # a second version to reclaim

    reader_conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    reader = await reader_conn.open_table(_Probe.TABLE_NAME)
    assert await reader.count_rows() == 6

    await repo.prune(dt.timedelta(seconds=0))

    assert await repo.count() == 6
    assert await reader.count_rows() == 6
    await _close(reader)
    await _close(reader_conn)
    await _close(table)
    await _close(writer_conn)


# ── path length ─────────────────────────────────────────────────────────────


async def test_deep_memory_root_still_stores_and_counts(tmp_path: Path) -> None:
    """Windows caps paths at 260 characters unless long paths are enabled.

    The memory root is user-chosen and Lance nests ``<table>.lance/data/<uuid>``
    and ``_indices/<uuid>/`` under it, so a deep root can cross the cap while
    looking perfectly ordinary. A red here on Windows means the store needs
    ``LongPathsEnabled`` and the install guide has to say so.
    """
    deep = tmp_path.joinpath(*(["x" * 40] * 4))  # +164 chars before Lance adds its own
    mr = MemoryRoot(deep)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table(_Probe.TABLE_NAME, schema=_Probe)
    await table.add([r.model_dump() for r in _rows(3)])
    assert await table.count_rows() == 3
    assert len(str(mr.lancedb_dir)) > 200, "probe must actually be deep"
    await _close(table)
    await _close(conn)
