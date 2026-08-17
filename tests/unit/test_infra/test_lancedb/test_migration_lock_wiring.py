"""LanceDB migration lock **wiring** — pins that the migration is
guarded by :func:`memory_root_lock`, NOT the flock semantics itself.

**Scope caveat (important for reviewers and future maintainers)**:
this module verifies the *wiring under test* — that
:func:`migrate_table_schemas` and :func:`migrate_fts_indexes` invoke
:func:`memory_root_lock` as their outer context manager, and that
the marker is re-checked inside the lock so a loser no-ops. To make
those assertions tractable in-process, every test here
``monkeypatch.setattr(lancedb_infra, "memory_root_lock", <fake>)``
with either a tracking async context manager or a single-process
``asyncio.Lock`` stand-in.

The ``fcntl.flock`` **semantics themselves** — multi-process
exclusion — cannot be exercised in-process (see the concurrent-tasks
test where this is called out inline). Real flock coverage lives in
``tests/unit/test_core/test_persistence/test_locking.py`` (spawns
subprocesses, exercises actual OS-level ``LOCK_EX`` semantics). If
:func:`memory_root_lock` ever changes implementation (e.g. dropping
``fcntl.flock`` for an in-process lock), that separate test suite is
what will regress — not this one.

Round-2 review finding M7: :func:`migrate_table_schemas` (and its
sibling :func:`migrate_fts_indexes`) were guarded only by an on-disk
marker file. Server startup and a concurrent ``everos cascade`` command
both call ``ensure_business_indexes`` → migration; both processes could
read the marker as 0, both proceed to mutate the schema, and races
could corrupt the on-disk state.

The fix wraps each migration in :func:`memory_root_lock` and re-checks
the marker *after* lock acquisition. This module pins that wiring:

* the lock is acquired around every migration invocation;
* the marker is re-checked inside the lock so a process that lost the
  race no-ops instead of re-running the migration;
* fail-loud semantics are preserved — a genuine ``alter_columns``
  failure still raises :class:`LanceDBMigrationError`, and the new
  message escalates from restart to wipe rather than jumping straight
  to a destructive recovery hint;
* two concurrent asyncio tasks serialize through the (faked) lock,
  demonstrating the intended sequencing behavior with the caveat
  that in-process ``asyncio.Lock`` is a proxy for the flock's
  ordering guarantee, not for the exclusion guarantee itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import everos.infra.persistence.lancedb as lancedb_infra
from everos.infra.persistence.lancedb import (
    _TABLE_SCHEMA_VERSION,
    Episode,
    LanceDBMigrationError,
    migrate_fts_indexes,
    migrate_table_schemas,
)


class _TrackingLock:
    """Async context manager that counts entries and exits.

    Stands in for :func:`memory_root_lock` in tests that need to observe
    lock acquisition without touching real ``fcntl`` state.
    """

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0
        self.active = 0
        self.max_active = 0

    def __call__(self, _memory_root: Any) -> AsyncIterator[None]:
        outer = self

        @asynccontextmanager
        async def _cm() -> AsyncIterator[None]:
            outer.entered += 1
            outer.active += 1
            outer.max_active = max(outer.max_active, outer.active)
            try:
                yield
            finally:
                outer.active -= 1
                outer.exited += 1

        return _cm()


def _serialising_lock_factory() -> tuple[
    Callable[[Any], AsyncIterator[None]], list[int]
]:
    """Return an async lock helper that serialises callers plus a
    ``max_concurrent`` list recording the peak in-flight count."""
    inner = asyncio.Lock()
    max_concurrent = [0]
    active = [0]

    @asynccontextmanager
    async def _cm(_memory_root: Any) -> AsyncIterator[None]:
        async with inner:
            active[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active[0])
            try:
                # Yield the event loop so a racer that already asked for
                # the lock gets a chance to run — proves serialization
                # rather than pure ordering.
                await asyncio.sleep(0)
                yield
            finally:
                active[0] -= 1

    return _cm, max_concurrent


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``MemoryRoot.resolve()`` at a fresh tmp path per test."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))


def _fake_table(nullable: bool, alter: Callable[..., Any] | None = None) -> Any:
    """Minimal table stub exposing the two coroutines the migration uses."""
    field = SimpleNamespace(nullable=nullable)
    arrow_schema = SimpleNamespace(field=lambda _name: field)

    async def _schema() -> Any:
        return arrow_schema

    async def _default_alter(*_alterations: Any) -> None:
        return None

    return SimpleNamespace(
        schema=_schema,
        alter_columns=alter if alter is not None else _default_alter,
    )


async def test_migrate_table_schemas_acquires_memory_root_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration must enter :func:`memory_root_lock` exactly once."""
    lock = _TrackingLock()
    monkeypatch.setattr(lancedb_infra, "memory_root_lock", lock)

    # Pre-write the marker to short-circuit inside the lock so we don't
    # exercise the LanceDB code path — we only care that the lock is
    # entered.
    marker = tmp_path / ".index" / "lancedb" / ".table_schema_version"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(_TABLE_SCHEMA_VERSION))

    await migrate_table_schemas()

    assert lock.entered == 1
    assert lock.exited == 1


async def test_migrate_table_schemas_rechecks_marker_after_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marker at target version → migration is a no-op even though the
    lock was acquired. No calls to ``get_table`` / ``alter_columns``."""
    lock = _TrackingLock()
    monkeypatch.setattr(lancedb_infra, "memory_root_lock", lock)

    marker = tmp_path / ".index" / "lancedb" / ".table_schema_version"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(_TABLE_SCHEMA_VERSION))

    get_table_calls: list[str] = []

    async def _get_table(name: str, _schema: Any) -> Any:
        get_table_calls.append(name)
        return _fake_table(nullable=True)

    monkeypatch.setattr(lancedb_infra, "get_table", _get_table)

    await migrate_table_schemas()

    assert lock.entered == 1
    assert get_table_calls == []


async def test_migrate_table_schemas_still_raises_on_alter_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-loud regression: a genuine ``alter_columns`` failure still
    raises :class:`LanceDBMigrationError`, and the message lists the
    escalating recovery steps in order (restart, then ``cascade rebuild``)."""
    lock = _TrackingLock()
    monkeypatch.setattr(lancedb_infra, "memory_root_lock", lock)

    async def _raise_alter(*_alterations: Any) -> None:
        raise RuntimeError("simulated alter_columns failure")

    call_count = {"n": 0}

    async def _get_table(name: str, _schema: Any) -> Any:
        call_count["n"] += 1
        if name == Episode.TABLE_NAME:
            return _fake_table(nullable=False, alter=_raise_alter)
        return _fake_table(nullable=True)

    monkeypatch.setattr(lancedb_infra, "get_table", _get_table)

    with pytest.raises(LanceDBMigrationError) as excinfo:
        await migrate_table_schemas()

    message = str(excinfo.value)
    assert Episode.TABLE_NAME in message
    restart_idx = message.find("restart the process")
    rebuild_idx = message.find("everos cascade rebuild")
    # Both hints present, restart first (escalating least- to most-destructive).
    assert restart_idx != -1
    assert rebuild_idx != -1
    assert restart_idx < rebuild_idx
    # The recovery must NOT be "delete the index dir": that leaves the cascade
    # queue marked done, so nothing re-indexes and the index comes back empty.
    assert "wipe the index directory" not in message
    assert "Do NOT just delete" in message

    # Marker must not be written on failure.
    marker = tmp_path / ".index" / "lancedb" / ".table_schema_version"
    assert not marker.exists()

    # Lock was still released cleanly.
    assert lock.entered == 1
    assert lock.exited == 1


async def test_migrate_fts_indexes_acquires_memory_root_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sibling migration is symmetric: wrapped in the same lock, and the
    marker re-check inside the lock short-circuits a duplicate run."""
    lock = _TrackingLock()
    monkeypatch.setattr(lancedb_infra, "memory_root_lock", lock)

    marker = tmp_path / ".index" / "lancedb" / ".fts_index_version"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(lancedb_infra._FTS_INDEX_SCHEMA_VERSION))

    get_table_calls: list[str] = []

    async def _get_table(name: str, _schema: Any) -> Any:
        get_table_calls.append(name)
        raise AssertionError("get_table must not be invoked in the no-op path")

    monkeypatch.setattr(lancedb_infra, "get_table", _get_table)

    await migrate_fts_indexes()

    assert lock.entered == 1
    assert get_table_calls == []


async def test_concurrent_migrations_serialize_through_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent asyncio tasks calling ``migrate_table_schemas``
    must not overlap the critical section. Uses an ``asyncio.Lock``
    stand-in for the process-level lock; ``fcntl.flock`` semantics can't
    be exercised inside a single Python process, but the wiring under
    test is identical — hold the lock around marker read + work +
    marker write."""
    lock_cm, max_concurrent = _serialising_lock_factory()
    monkeypatch.setattr(lancedb_infra, "memory_root_lock", lock_cm)

    # Track how many tasks actually reached the work section (i.e. saw
    # marker < target). Exactly one should; the other must see the
    # marker written by the winner and no-op.
    reached_work = {"n": 0}

    async def _get_table(_name: str, _schema: Any) -> Any:
        reached_work["n"] += 1
        # Give the other task a chance to try acquiring the lock while
        # this one holds it — reinforces the serialization assertion.
        await asyncio.sleep(0)
        return _fake_table(nullable=True)

    monkeypatch.setattr(lancedb_infra, "get_table", _get_table)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(migrate_table_schemas())
        tg.create_task(migrate_table_schemas())

    assert max_concurrent[0] == 1
    # One task wrote the marker; the other saw it and short-circuited.
    assert reached_work["n"] == len(lancedb_infra.BUSINESS_SCHEMAS_WITH_VECTOR)
    marker = tmp_path / ".index" / "lancedb" / ".table_schema_version"
    assert int(marker.read_text().strip()) == _TABLE_SCHEMA_VERSION
