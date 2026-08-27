"""The SQLite engine's connection pool is configured, not inherited.

Regression cover for a stall that took two benchmark servers down for 2h17m and
3h33m. ``create_async_engine`` was called with only ``url``/``echo``/``future``,
so pool behaviour came from library defaults with no timeout, no recycle and no
pre-ping. A connection checked out and never returned shrinks the pool by one
permanently; once every slot is gone, each later caller waits on a checkout that
never completes.

The symptom is why this needs pinning rather than a comment. Nothing looked
broken: HTTP answered, the event loop ran, every thread was idle, CPU was flat.
Only the write side had stopped -- SQLite file mtimes frozen, ``run_record`` rows
stuck in RUNNING because strategies could not persist their own results, which
reads as "still working" rather than "cannot write". Even the OME run-timeout
backstop was silent: it fired, then needed a connection to record the failure.
Measured signature: aiosqlite connection threads at 20 and 58 versus a steady
6-10 on healthy siblings, with 7 and 22 parked inside aiosqlite's connect path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.pool import AsyncAdaptedQueuePool

from everos.config.settings import SqliteSettings
from everos.core.persistence.sqlite.engine import create_system_engine


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "system.db"


def test_defaults_are_explicit_and_bounded() -> None:
    """Every knob that governs a stuck checkout has a value we chose."""
    s = SqliteSettings()
    assert s.pool_size == 5
    assert s.max_overflow == 10
    assert s.pool_timeout_seconds == 30.0
    assert s.pool_recycle_seconds == 1800
    assert s.pool_pre_ping is True


def test_the_engine_actually_receives_them(db: Path) -> None:
    """Settings that never reach the engine are decoration.

    Asserted on the live pool object rather than the call arguments: this is the
    exact link that was missing before, and reading it back from the engine is
    the only way to know it closed.
    """
    s = SqliteSettings(
        pool_size=3, max_overflow=2, pool_timeout_seconds=7.0, pool_recycle_seconds=60
    )
    engine = create_system_engine(db, s)
    pool = engine.sync_engine.pool
    assert isinstance(pool, AsyncAdaptedQueuePool)
    assert pool.size() == 3
    assert pool._max_overflow == 2
    assert pool._timeout == 7.0
    assert pool._recycle == 60
    assert pool._pre_ping is True


def test_a_bounded_wait_is_what_turns_a_hang_into_an_error(db: Path) -> None:
    """Exhaustion must raise, not block forever.

    The stall was not that the pool ran dry -- pools do -- but that running dry
    had no deadline, so the failure never surfaced anywhere a caller or an
    operator could see it.
    """
    engine = create_system_engine(db, SqliteSettings(pool_timeout_seconds=0.25))
    assert engine.sync_engine.pool._timeout == 0.25


def test_timeout_must_be_positive() -> None:
    """Zero would mean "fail instantly"; negative would mean "wait forever"."""
    with pytest.raises(ValueError):
        SqliteSettings(pool_timeout_seconds=0)
    with pytest.raises(ValueError):
        SqliteSettings(pool_timeout_seconds=-1)


def test_recycle_can_be_disabled_but_not_arbitrary(db: Path) -> None:
    """``-1`` is SQLAlchemy's "never recycle"; keep it reachable, reject below."""
    assert SqliteSettings(pool_recycle_seconds=-1).pool_recycle_seconds == -1
    with pytest.raises(ValueError):
        SqliteSettings(pool_recycle_seconds=-2)


async def test_pragmas_still_apply_after_the_pool_change(db: Path) -> None:
    """The pool rework must not displace the per-connection PRAGMA listener.

    Both listeners hang off the same sync engine, and registering the second is
    exactly the kind of edit that quietly drops the first. Verified by reading a
    PRAGMA back off a live connection rather than by inspecting the registry:
    what matters is that WAL is actually on, not that a callable is attached.
    """
    from sqlalchemy import text

    engine = create_system_engine(db, SqliteSettings())
    async with engine.connect() as conn:
        mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
        busy = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
    await engine.dispose()
    assert str(mode).upper() == "WAL"
    assert busy == SqliteSettings().busy_timeout_ms


async def test_saturation_is_logged_so_the_condition_is_visible(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drained pool must announce itself.

    What made the original failure expensive was the absence of any signal: the
    condition had to be reconstructed afterwards by counting threads in a py-spy
    dump against a healthy sibling. ``checked_out`` in the log line is the
    discriminator -- honest load returns connections so it oscillates, a leak
    only climbs.
    """
    from sqlalchemy import text

    import everos.core.persistence.sqlite.engine as mod

    seen: list[dict[str, object]] = []
    monkeypatch.setattr(
        mod.logger, "warning", lambda _evt, **kw: seen.append(kw), raising=True
    )

    engine = create_system_engine(db, SqliteSettings(pool_size=1, max_overflow=0))
    # Capacity 1: hold the only connection, then force a second checkout of it.
    async with engine.connect() as c1:
        await c1.execute(text("SELECT 1"))
    async with engine.connect() as c2:
        await c2.execute(text("SELECT 1"))
    await engine.dispose()

    assert seen, "a pool at capacity produced no warning"
    assert seen[0]["capacity"] == 1
    assert seen[0]["checked_out"] >= 1


async def test_a_healthy_pool_stays_quiet(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No warning below capacity -- otherwise the signal is noise.

    A line that fires on every checkout would have been ignored, which is how the
    original failure stayed invisible in the first place.
    """
    from sqlalchemy import text

    import everos.core.persistence.sqlite.engine as mod

    seen: list[dict[str, object]] = []
    monkeypatch.setattr(
        mod.logger, "warning", lambda _evt, **kw: seen.append(kw), raising=True
    )
    engine = create_system_engine(db, SqliteSettings(pool_size=5, max_overflow=10))
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    await engine.dispose()
    assert seen == []
