"""Cross-process exclusion contracts for LanceDB projection lifecycle locks."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import portalocker
import pytest

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb.projection_lock import (
    ProjectionLockUnavailableError,
    ome_lock_path,
    projection_lock_path,
    projection_rebuild_lock,
    projection_server_lock,
)

_LOCK_HOLDER = """
import portalocker
import sys

path = sys.argv[1]
flags = int(sys.argv[2]) | portalocker.LOCK_NB
handle = open(path, "a+")
portalocker.lock(handle, flags)
print("LOCKED", flush=True)
sys.stdin.readline()
portalocker.unlock(handle)
handle.close()
"""


@contextmanager
def _external_lock(path: Path, flags: int) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, str(path), str(flags)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    ready = process.stdout.readline().strip()
    if ready != "LOCKED":
        _out, err = process.communicate(timeout=5)
        raise AssertionError(f"external lock helper failed: {err}")
    try:
        yield
    finally:
        if process.stdin is not None:
            process.stdin.write("\n")
            process.stdin.flush()
        process.communicate(timeout=5)


async def test_external_server_shared_lock_allows_shared_and_blocks_rebuild(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)

    with _external_lock(projection_lock_path(root), int(portalocker.LOCK_SH)):
        async with projection_server_lock(root, timeout_seconds=0.2):
            pass
        with pytest.raises(ProjectionLockUnavailableError, match="projection"):
            async with projection_rebuild_lock(root):
                raise AssertionError("exclusive rebuild lock must not be entered")

    async with projection_rebuild_lock(root):
        pass


async def test_external_rebuild_exclusive_lock_blocks_server_startup(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)

    with (
        _external_lock(projection_lock_path(root), int(portalocker.LOCK_EX)),
        pytest.raises(ProjectionLockUnavailableError, match="timed out"),
    ):
        async with projection_server_lock(root, timeout_seconds=0.01):
            raise AssertionError("server lock entered during active rebuild")

    async with projection_server_lock(root, timeout_seconds=0.2):
        pass


async def test_external_legacy_ome_lock_blocks_rebuild_before_body(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)
    entered = False

    with (
        _external_lock(ome_lock_path(root), int(portalocker.LOCK_EX)),
        pytest.raises(ProjectionLockUnavailableError, match="OME"),
    ):
        async with projection_rebuild_lock(root):
            entered = True

    assert entered is False
    async with projection_rebuild_lock(root):
        pass
