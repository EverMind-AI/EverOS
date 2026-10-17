"""Unit tests for the LanceDB async connection factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.config import LanceDBSettings
from everos.core.persistence import MemoryRoot, open_lancedb_connection


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


async def test_connect_creates_lancedb_dir(memory_root: MemoryRoot) -> None:
    settings = LanceDBSettings()
    # Remove the auto-created dir to verify the factory recreates it.
    if memory_root.lancedb_dir.exists():
        memory_root.lancedb_dir.rmdir()
    assert not memory_root.lancedb_dir.exists()

    conn = await open_lancedb_connection(memory_root.lancedb_dir, settings)
    try:
        assert memory_root.lancedb_dir.is_dir()
        assert conn.is_open()
    finally:
        conn.close()  # AsyncConnection.close() is sync


async def test_empty_connection_lists_no_tables(memory_root: MemoryRoot) -> None:
    settings = LanceDBSettings()
    conn = await open_lancedb_connection(memory_root.lancedb_dir, settings)
    try:
        # list_tables() returns ListTablesResponse(tables, page_token).
        result = await conn.list_tables()
        assert list(result.tables) == []
    finally:
        conn.close()


async def test_read_consistency_seconds_translated_to_timedelta(
    memory_root: MemoryRoot,
) -> None:
    """Non-None read_consistency_seconds must be passed as a timedelta."""
    settings = LanceDBSettings(read_consistency_seconds=5.0)
    conn = await open_lancedb_connection(memory_root.lancedb_dir, settings)
    try:
        # The interval echoed back from the connection should equal what we set.
        # AsyncConnection.get_read_consistency_interval is async.
        import datetime as dt

        interval = await conn.get_read_consistency_interval()
        assert interval == dt.timedelta(seconds=5.0)
    finally:
        conn.close()


async def test_default_consistency_is_none(memory_root: MemoryRoot) -> None:
    settings = LanceDBSettings()
    conn = await open_lancedb_connection(memory_root.lancedb_dir, settings)
    try:
        interval = await conn.get_read_consistency_interval()
        assert interval is None
    finally:
        conn.close()


async def test_index_cache_cap_is_plumbed_into_session(
    memory_root: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capped ``Session`` must reach ``lancedb.connect_async``.

    The connection factory's whole purpose for installing a Session is
    to bound the index reader cache so FDs do not leak. We spy on the
    underlying ``connect_async`` and assert a Session is passed —
    Session objects don't expose the configured cap back as a property,
    so verifying that a Session is wired through is the closest unit-
    level check we can make. The behavioural side (LRU eviction →
    FD release under load) is covered by the fd-probe scripts kept
    outside the test suite.
    """
    import lancedb

    settings = LanceDBSettings(
        index_cache_size_bytes=1024, metadata_cache_size_bytes=2048
    )
    captured: dict[str, object] = {}

    real_connect = lancedb.connect_async

    async def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["session"] = kwargs.get("session")
        return await real_connect(*args, **kwargs)

    monkeypatch.setattr(lancedb, "connect_async", spy)

    conn = await open_lancedb_connection(memory_root.lancedb_dir, settings)
    try:
        assert isinstance(captured.get("session"), lancedb.Session)
    finally:
        conn.close()


async def test_metadata_cache_is_bounded(memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The metadata cache must be capped, not left at lancedb's unbounded
    default.

    Lance keys metadata entries by dataset version and does not evict them
    when cleanup deletes the underlying files (lance-format/lance#7718), so
    each surviving entry pins its index reader's descriptors for the life of
    the process. Measured: unbounded grows linearly past 600 descriptors over
    80 optimize rounds with no plateau, while a 1 MB cap converges by round 31.
    A regression that reinstates ``None`` here is the one that brings the leak
    back, so pin the argument rather than only the Session's presence.
    """
    import lancedb

    settings = LanceDBSettings(metadata_cache_size_bytes=4096)
    captured: dict[str, object] = {}
    real_connect = lancedb.connect_async

    async def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        session = kwargs.get("session")
        captured["metadata_cap"] = getattr(session, "_everos_metadata_cap", "unknown")
        captured["session"] = session
        return await real_connect(*args, **kwargs)

    real_session = lancedb.Session

    def session_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["metadata_cache_size_bytes"] = kwargs.get(
            "metadata_cache_size_bytes", "absent"
        )
        return real_session(*args, **kwargs)

    monkeypatch.setattr(lancedb, "connect_async", spy)
    monkeypatch.setattr(lancedb, "Session", session_spy)

    conn = await open_lancedb_connection(memory_root.lancedb_dir, settings)
    try:
        assert captured["metadata_cache_size_bytes"] == 4096, (
            "metadata cache was not capped from settings: "
            f"{captured['metadata_cache_size_bytes']!r}"
        )
    finally:
        conn.close()


def test_metadata_cache_default_is_bounded() -> None:
    """The shipped default must be a real cap, not ``None``."""
    cap = LanceDBSettings().metadata_cache_size_bytes
    assert isinstance(cap, int) and cap > 0, f"metadata cache default is {cap!r}"
