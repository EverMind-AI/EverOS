"""Storage-generation gates around mutating cascade CLI commands."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from typer.testing import CliRunner

from everos.config import load_settings
from everos.core.persistence import MemoryRoot
from everos.entrypoints.cli.commands import cascade as cascade_mod
from everos.infra.persistence.lancedb.projection_lock import (
    ProjectionLockUnavailableError,
    projection_rebuild_lock,
)
from everos.infra.persistence.lancedb.storage_identity import (
    StorageIdentityMigrationRequiredError,
    ensure_storage_identity_ready,
    mark_storage_identity_ready,
    mark_storage_identity_rebuilding,
    marker_path,
    read_storage_identity_state,
)

_MUTATING_RUNTIME_HOLDER = """
import asyncio
import os
import sys

os.environ["EVEROS_ROOT"] = sys.argv[1]

from everos.entrypoints.cli.commands.cascade import _runtime

async def main():
    async with _runtime(verify=False, ensure=False):
        print("LOCKED", flush=True)
        sys.stdin.readline()

asyncio.run(main())
"""


@contextmanager
def _external_mutating_runtime(root: Path) -> Iterator[None]:
    process = subprocess.Popen(
        [sys.executable, "-c", _MUTATING_RUNTIME_HOLDER, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    startup_lines: list[str] = []
    while True:
        ready = process.stdout.readline()
        if not ready:
            break
        startup_lines.append(ready.rstrip())
        if ready.strip() == "LOCKED":
            break
    if not startup_lines or startup_lines[-1] != "LOCKED":
        _out, err = process.communicate(timeout=10)
        raise AssertionError(
            f"mutating runtime helper failed: stdout={startup_lines!r}, stderr={err!r}"
        )
    try:
        yield
    finally:
        if process.stdin is not None:
            process.stdin.write("\n")
            process.stdin.flush()
        process.communicate(timeout=10)


def _set_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("EVEROS_ROOT", str(root))
    load_settings.cache_clear()


def test_sync_is_blocked_while_rebuild_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    mark_storage_identity_rebuilding(MemoryRoot(tmp_path))

    result = CliRunner().invoke(cascade_mod.app, ["sync"])

    assert result.exit_code != 0
    assert isinstance(result.exception, StorageIdentityMigrationRequiredError)


def test_status_remains_available_without_opening_lancedb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    mark_storage_identity_rebuilding(MemoryRoot(tmp_path))

    async def forbidden_connection():  # type: ignore[no-untyped-def]
        raise AssertionError("read-only status must not open LanceDB")

    monkeypatch.setattr(cascade_mod, "get_connection", forbidden_connection)

    result = CliRunner().invoke(cascade_mod.app, ["status"])

    assert result.exit_code == 0, result.output
    assert "pending:" in result.output


async def test_runtime_initialization_failure_disposes_under_shared_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed CLI startup closes both stores before releasing exclusion."""
    _set_root(monkeypatch, tmp_path)
    root = MemoryRoot(tmp_path)
    mark_storage_identity_ready(root)
    disposed: list[str] = []

    async def fail_after_open() -> None:
        raise RuntimeError("schema verification failed")

    async def dispose_lance() -> None:
        with pytest.raises(ProjectionLockUnavailableError, match="projection"):
            async with projection_rebuild_lock(root):
                raise AssertionError("rebuild entered before LanceDB cleanup")
        disposed.append("lancedb")
        raise RuntimeError("LanceDB cleanup also failed")

    async def dispose_sqlite() -> None:
        with pytest.raises(ProjectionLockUnavailableError, match="projection"):
            async with projection_rebuild_lock(root):
                raise AssertionError("rebuild entered before SQLite cleanup")
        disposed.append("sqlite")

    monkeypatch.setattr(cascade_mod, "verify_business_schemas", fail_after_open)
    monkeypatch.setattr(cascade_mod, "dispose_connection", dispose_lance)
    monkeypatch.setattr(cascade_mod, "dispose_engine", dispose_sqlite)

    with pytest.raises(RuntimeError, match="schema verification failed"):
        async with cascade_mod._runtime():
            raise AssertionError("runtime body must not be entered")

    assert disposed == ["lancedb", "lancedb", "sqlite"]
    async with projection_rebuild_lock(root):
        pass


async def test_runtime_shutdown_attempts_both_disposals_when_first_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steady-state teardown must not strand SQLite after a LanceDB failure."""
    _set_root(monkeypatch, tmp_path)
    mark_storage_identity_ready(MemoryRoot(tmp_path))
    real_dispose_lancedb = cascade_mod.dispose_connection
    real_dispose_sqlite = cascade_mod.dispose_engine
    disposed: list[str] = []

    async def fail_lancedb_disposal() -> None:
        disposed.append("lancedb")
        await real_dispose_lancedb()
        raise RuntimeError("LanceDB cleanup failed")

    async def dispose_sqlite() -> None:
        disposed.append("sqlite")
        await real_dispose_sqlite()

    monkeypatch.setattr(cascade_mod, "dispose_connection", fail_lancedb_disposal)
    monkeypatch.setattr(cascade_mod, "dispose_engine", dispose_sqlite)

    with pytest.raises(RuntimeError, match="LanceDB cleanup failed"):
        async with cascade_mod._runtime(verify=False, ensure=False):
            pass

    assert disposed == ["lancedb", "lancedb", "sqlite"]


async def test_runtime_shutdown_cancellation_keeps_rebuild_excluded_until_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    root = MemoryRoot(tmp_path)
    mark_storage_identity_ready(root)
    real_dispose_lancedb = cascade_mod.dispose_connection
    dispose_started = anyio.Event()
    allow_dispose = anyio.Event()
    runtime_done = anyio.Event()
    cancel_scope: anyio.CancelScope | None = None

    async def delayed_dispose() -> None:
        dispose_started.set()
        await allow_dispose.wait()
        await real_dispose_lancedb()

    monkeypatch.setattr(cascade_mod, "dispose_connection", delayed_dispose)

    async def run_runtime() -> None:
        nonlocal cancel_scope
        with anyio.CancelScope() as scope:
            cancel_scope = scope
            try:
                async with cascade_mod._runtime(verify=False, ensure=False):
                    pass
            finally:
                runtime_done.set()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run_runtime)
        await dispose_started.wait()
        assert cancel_scope is not None
        cancel_scope.cancel()
        await anyio.lowlevel.checkpoint()

        assert not runtime_done.is_set()
        with pytest.raises(ProjectionLockUnavailableError):
            async with projection_rebuild_lock(root):
                raise AssertionError("rebuild entered during shielded cleanup")

        allow_dispose.set()
        await runtime_done.wait()

    async with projection_rebuild_lock(root):
        pass


async def test_runtime_uses_shared_then_exclusive_bootstrap_lock_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    events: list[str] = []

    @asynccontextmanager
    async def server_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        events.append("projection_shared_enter")
        try:
            yield
        finally:
            events.append("projection_shared_exit")

    @asynccontextmanager
    async def bootstrap_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        assert events == ["projection_shared_enter"]
        events.append("bootstrap_exclusive_enter")
        try:
            yield
        finally:
            events.append("bootstrap_exclusive_exit")

    class FakeConnection:
        async def run_sync(self, _operation) -> None:  # type: ignore[no-untyped-def]
            events.append("sqlite_schema")

    class FakeBegin:
        async def __aenter__(self) -> FakeConnection:
            return FakeConnection()

        async def __aexit__(self, *_args) -> None:  # type: ignore[no-untyped-def]
            return None

    class FakeEngine:
        def begin(self) -> FakeBegin:
            return FakeBegin()

    async def record(name: str) -> None:
        events.append(name)

    def get_engine() -> FakeEngine:
        events.append("sqlite_engine")
        return FakeEngine()

    monkeypatch.setattr(cascade_mod, "projection_server_lock", server_lock)
    monkeypatch.setattr(cascade_mod, "projection_bootstrap_lock", bootstrap_lock)
    monkeypatch.setattr(
        cascade_mod,
        "verify_storage_identity_ready",
        lambda: record("identity_gate"),
    )
    monkeypatch.setattr(cascade_mod, "get_engine", get_engine)
    monkeypatch.setattr(cascade_mod, "get_connection", lambda: record("lancedb"))
    monkeypatch.setattr(
        cascade_mod,
        "verify_business_schemas",
        lambda: record("schema_verify"),
    )
    monkeypatch.setattr(
        cascade_mod,
        "ensure_business_indexes",
        lambda: record("index_ensure"),
    )
    monkeypatch.setattr(
        cascade_mod,
        "dispose_connection",
        lambda: record("dispose_lancedb"),
    )
    monkeypatch.setattr(
        cascade_mod,
        "dispose_engine",
        lambda: record("dispose_sqlite"),
    )

    async with cascade_mod._runtime():
        events.append("runtime_body")

    assert events == [
        "projection_shared_enter",
        "bootstrap_exclusive_enter",
        "identity_gate",
        "sqlite_engine",
        "sqlite_schema",
        "lancedb",
        "schema_verify",
        "index_ensure",
        "bootstrap_exclusive_exit",
        "runtime_body",
        "dispose_lancedb",
        "dispose_sqlite",
        "projection_shared_exit",
    ]


async def test_rebuild_runtime_bypasses_both_lifecycle_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    events: list[str] = []

    @asynccontextmanager
    async def forbidden_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        raise AssertionError("rebuild runtime must not acquire lifecycle locks")
        yield

    class FakeConnection:
        async def run_sync(self, _operation) -> None:  # type: ignore[no-untyped-def]
            events.append("sqlite_schema")

    class FakeBegin:
        async def __aenter__(self) -> FakeConnection:
            return FakeConnection()

        async def __aexit__(self, *_args) -> None:  # type: ignore[no-untyped-def]
            return None

    class FakeEngine:
        def begin(self) -> FakeBegin:
            return FakeBegin()

    monkeypatch.setattr(cascade_mod, "projection_server_lock", forbidden_lock)
    monkeypatch.setattr(cascade_mod, "projection_bootstrap_lock", forbidden_lock)
    monkeypatch.setattr(cascade_mod, "get_engine", FakeEngine)

    async def record(name: str) -> None:
        events.append(name)

    monkeypatch.setattr(cascade_mod, "get_connection", lambda: record("lancedb"))
    monkeypatch.setattr(
        cascade_mod,
        "dispose_connection",
        lambda: record("dispose_lancedb"),
    )
    monkeypatch.setattr(
        cascade_mod,
        "dispose_engine",
        lambda: record("dispose_sqlite"),
    )

    async with cascade_mod._runtime(
        verify=False,
        ensure=False,
        identity_gate=False,
        lifecycle_lock=False,
    ):
        events.append("runtime_body")

    assert events == [
        "sqlite_schema",
        "lancedb",
        "runtime_body",
        "dispose_lancedb",
        "dispose_sqlite",
    ]


def test_rebuild_sets_rebuilding_before_runtime_and_ready_only_after_clean_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    root = MemoryRoot(tmp_path)

    @asynccontextmanager
    async def fake_runtime(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {
            "verify": False,
            "ensure": False,
            "identity_gate": False,
            "lifecycle_lock": False,
        }
        state = read_storage_identity_state(root)
        assert state is not None and state.state == "REBUILDING"
        yield

    class FakeOrchestrator:
        async def sync_once(self) -> int:
            return 7

        async def drain_once(self) -> int:
            return 0

        async def queue_summary(self):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                pending=0,
                failed_retryable=0,
                failed_permanent=0,
            )

    async def zero() -> int:
        return 0

    async def no_tables() -> list[str]:
        return []

    async def noop() -> None:
        return None

    monkeypatch.setattr(cascade_mod, "_runtime", fake_runtime)
    monkeypatch.setattr(cascade_mod.md_change_state_repo, "reset_all", zero)
    monkeypatch.setattr(cascade_mod, "drop_business_tables", no_tables)
    monkeypatch.setattr(cascade_mod, "ensure_business_indexes", noop)
    monkeypatch.setattr(cascade_mod, "_build_orchestrator", FakeOrchestrator)

    result = CliRunner().invoke(cascade_mod.app, ["rebuild", "--yes"])

    assert result.exit_code == 0, result.output
    assert "rebuild complete" in result.output
    state = read_storage_identity_state(root)
    assert state is not None
    assert state.state == "READY"
    assert state.generation == 2


def test_failed_rebuild_leaves_rebuilding_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    root = MemoryRoot(tmp_path)

    @asynccontextmanager
    async def fake_runtime(**kwargs):  # type: ignore[no-untyped-def]
        yield

    class FailedOrchestrator:
        async def sync_once(self) -> int:
            return 1

        async def drain_once(self) -> int:
            return 0

        async def queue_summary(self):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                pending=0,
                failed_retryable=0,
                failed_permanent=1,
            )

    async def zero() -> int:
        return 0

    async def no_tables() -> list[str]:
        return []

    async def noop() -> None:
        return None

    monkeypatch.setattr(cascade_mod, "_runtime", fake_runtime)
    monkeypatch.setattr(cascade_mod.md_change_state_repo, "reset_all", zero)
    monkeypatch.setattr(cascade_mod, "drop_business_tables", no_tables)
    monkeypatch.setattr(cascade_mod, "ensure_business_indexes", noop)
    monkeypatch.setattr(cascade_mod, "_build_orchestrator", FailedOrchestrator)

    result = CliRunner().invoke(cascade_mod.app, ["rebuild", "--yes"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "failed_permanent=1" in str(result.exception)
    state = read_storage_identity_state(root)
    assert state is not None and state.state == "REBUILDING"


def test_interrupted_rebuild_leaves_rebuilding_and_blocks_normal_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    root = MemoryRoot(tmp_path)

    @asynccontextmanager
    async def fake_runtime(**kwargs):  # type: ignore[no-untyped-def]
        state = read_storage_identity_state(root)
        assert state is not None and state.state == "REBUILDING"
        yield

    async def zero() -> int:
        return 0

    async def interrupt_after_queue_reset() -> list[str]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cascade_mod, "_runtime", fake_runtime)
    monkeypatch.setattr(cascade_mod.md_change_state_repo, "reset_all", zero)
    monkeypatch.setattr(
        cascade_mod, "drop_business_tables", interrupt_after_queue_reset
    )

    result = CliRunner().invoke(cascade_mod.app, ["rebuild", "--yes"])

    assert result.exit_code == 130
    assert "remains REBUILDING" in result.stderr
    state = read_storage_identity_state(root)
    assert state is not None and state.state == "REBUILDING"
    with pytest.raises(StorageIdentityMigrationRequiredError, match="REBUILDING"):
        ensure_storage_identity_ready(root)


def test_rebuild_lock_refusal_happens_before_generation_or_index_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_root(monkeypatch, tmp_path)
    root = MemoryRoot(tmp_path)
    mutation_called = False

    @asynccontextmanager
    async def unavailable(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        raise ProjectionLockUnavailableError("projection exclusive lock is held")
        yield

    async def forbidden_drop() -> list[str]:
        nonlocal mutation_called
        mutation_called = True
        return []

    monkeypatch.setattr(cascade_mod, "projection_rebuild_lock", unavailable)
    monkeypatch.setattr(cascade_mod, "drop_business_tables", forbidden_drop)

    result = CliRunner().invoke(cascade_mod.app, ["rebuild", "--yes"])

    assert result.exit_code == 3
    assert "did not modify" in result.stderr
    assert mutation_called is False
    assert not marker_path(root).exists()


async def test_mutating_cli_runtime_excludes_rebuild_across_processes(
    tmp_path: Path,
) -> None:
    """A CLI writer holds the shared lifecycle lock until its stores close."""
    root = MemoryRoot(tmp_path)
    mark_storage_identity_ready(root)
    with _external_mutating_runtime(tmp_path):
        with pytest.raises(ProjectionLockUnavailableError, match="projection"):
            async with projection_rebuild_lock(root):
                raise AssertionError("rebuild entered while CLI writer was active")
        state = read_storage_identity_state(root)
        assert state is not None and state.state == "READY"

    async with projection_rebuild_lock(root):
        pass
