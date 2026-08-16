"""Cross-process exclusion contracts for LanceDB projection lifecycle locks."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import portalocker
import pytest

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb.projection_lock import (
    ProjectionLockUnavailableError,
    ome_lock_path,
    projection_bootstrap_lock,
    projection_bootstrap_lock_path,
    projection_lock_path,
    projection_rebuild_lock,
    projection_server_lock,
)
from everos.infra.persistence.lancedb.storage_identity import (
    read_storage_identity_state,
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

_FRESH_BOOTSTRAP_WORKER = """
import asyncio
import sys
from pathlib import Path

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb.projection_lock import (
    projection_bootstrap_lock,
    projection_server_lock,
)
from everos.infra.persistence.lancedb.storage_identity import (
    ensure_storage_identity_ready,
)

root = MemoryRoot(Path(sys.argv[1]))
role = sys.argv[2]
entered_path = Path(sys.argv[3])

async def main():
    async with projection_server_lock(root):
        print(f"{role}:SHARED", flush=True)
        async with projection_bootstrap_lock(root):
            entered_path.write_text("entered", encoding="utf-8")
            print(f"{role}:BOOTSTRAP", flush=True)
            if role == "first":
                sys.stdin.readline()
            ensure_storage_identity_ready(root)
            print(f"{role}:READY", flush=True)

asyncio.run(main())
"""

_FRESH_PROVIDER_WORKER = """
import asyncio
import os
import sys
import time
from pathlib import Path

os.environ["EVEROS_ROOT"] = sys.argv[1]

from fastapi import FastAPI

from everos.config import load_settings
from everos.entrypoints.api.lifespans.lancedb import LanceDBLifespanProvider

load_settings.cache_clear()
ready_path = Path(sys.argv[2])
go_path = Path(sys.argv[3])
ready_path.write_text("ready", encoding="utf-8")
while not go_path.exists():
    time.sleep(0.01)

async def main():
    provider = LanceDBLifespanProvider()
    app = FastAPI()
    await provider.startup(app)
    await provider.shutdown(app)

asyncio.run(main())
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


async def test_bootstrap_lock_is_exclusive_beneath_shared_projection_lock(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)
    assert projection_bootstrap_lock_path(root) == (
        root.index_dir / ".projection.bootstrap.lock"
    )

    async with projection_server_lock(root, timeout_seconds=0.2):
        with (
            _external_lock(
                projection_bootstrap_lock_path(root), int(portalocker.LOCK_EX)
            ),
            pytest.raises(ProjectionLockUnavailableError, match="bootstrap"),
        ):
            async with projection_bootstrap_lock(root, timeout_seconds=0.01):
                raise AssertionError("concurrent bootstrap lock must not be entered")

        with pytest.raises(RuntimeError, match="bootstrap failed"):
            async with projection_bootstrap_lock(root, timeout_seconds=0.2):
                raise RuntimeError("bootstrap failed")

        # A failed bootstrap body releases the exclusive side for a retry while
        # the caller continues to retain its shared lifecycle exclusion.
        async with projection_bootstrap_lock(root, timeout_seconds=0.2):
            pass


def test_fresh_initialization_is_serialized_across_processes(tmp_path: Path) -> None:
    """A second fresh process waits and observes the first READY marker."""
    first_entered = tmp_path / "first-entered"
    second_entered = tmp_path / "second-entered"

    def start(role: str, entered_path: Path) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                _FRESH_BOOTSTRAP_WORKER,
                str(tmp_path),
                role,
                str(entered_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    first = start("first", first_entered)
    second: subprocess.Popen[str] | None = None
    try:
        assert first.stdout is not None
        assert first.stdout.readline().strip() == "first:SHARED"
        assert first.stdout.readline().strip() == "first:BOOTSTRAP"
        assert first_entered.exists()

        second = start("second", second_entered)
        assert second.stdout is not None
        assert second.stdout.readline().strip() == "second:SHARED"

        # The first process has conclusively entered the exclusive bootstrap
        # section. Give the second process multiple polling intervals; it must
        # remain outside until the first publishes READY and releases the lock.
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline and not second_entered.exists():
            time.sleep(0.01)
        assert not second_entered.exists()
        assert second.poll() is None

        assert first.stdin is not None
        first.stdin.write("\n")
        first.stdin.flush()
        first_out, first_err = first.communicate(timeout=10)
        second_out, second_err = second.communicate(timeout=10)

        assert first.returncode == 0, f"stdout={first_out!r}, stderr={first_err!r}"
        assert second.returncode == 0, f"stdout={second_out!r}, stderr={second_err!r}"
        assert "first:READY" in first_out
        assert "second:BOOTSTRAP" in second_out
        assert "second:READY" in second_out
        assert second_entered.exists()
        state = read_storage_identity_state(MemoryRoot(tmp_path))
        assert state is not None
        assert state.generation == 2
        assert state.state == "READY"
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)


def test_fresh_provider_startup_is_serialized_across_processes(
    tmp_path: Path,
) -> None:
    """Real marker, table, and index bootstrap succeeds for concurrent APIs."""
    memory_path = tmp_path / "memory"
    coordination_path = tmp_path / "coordination"
    coordination_path.mkdir()
    go_path = coordination_path / "go"
    processes: list[subprocess.Popen[str]] = []

    try:
        for index in range(4):
            ready_path = coordination_path / f"ready-{index}"
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _FRESH_PROVIDER_WORKER,
                        str(memory_path),
                        str(ready_path),
                        str(go_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if len(list(coordination_path.glob("ready-*"))) == len(processes):
                break
            assert all(process.poll() is None for process in processes)
            time.sleep(0.02)
        else:
            raise AssertionError("provider workers did not reach the start barrier")

        go_path.write_text("go", encoding="utf-8")
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, (
                f"returncode={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
            )

        memory_root = MemoryRoot(memory_path)
        state = read_storage_identity_state(memory_root)
        assert state is not None
        assert state.generation == 2
        assert state.state == "READY"
        assert len(list(memory_root.lancedb_dir.glob("*.lance"))) == 7
        assert not list(memory_root.lancedb_dir.glob("..storage_identity.json.*"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)


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
