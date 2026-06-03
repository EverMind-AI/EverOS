"""Async LanceDB connection factory.

LanceDB does not live inside the SQLAlchemy ecosystem; it has its own
``connect_async`` returning :class:`lancedb.AsyncConnection`. This module
is a thin wrapper that:

    1. ensures the lancedb root directory exists
    2. converts ``LanceDBSettings.read_consistency_seconds`` into the
       :class:`datetime.timedelta` value LanceDB expects
    3. installs a capped :class:`lancedb.Session` so the global index
       cache cannot grow unbounded and exhaust file descriptors
       (see :attr:`LanceDBSettings.index_cache_size_bytes` for the
       full rationale)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import lancedb
from lancedb import AsyncConnection

from everos.config import LanceDBSettings


async def open_lancedb_connection(
    lancedb_dir: Path,
    lancedb_settings: LanceDBSettings,
) -> AsyncConnection:
    """Open an async LanceDB connection rooted at ``lancedb_dir``.

    Args:
        lancedb_dir: Filesystem path to the LanceDB root (typically
            ``MemoryRoot.lancedb_dir``). Created if missing.
        lancedb_settings: Tunables; the ``read_consistency_seconds`` field
            is converted to a :class:`~datetime.timedelta`, and
            ``index_cache_size_bytes`` caps the global index cache.

    Returns:
        An :class:`AsyncConnection` ready for table operations.
    """
    # mkdir is a microsecond-fast syscall and only fires on first connect;
    # not worth pulling in anyio.Path / aiofiles for it.
    lancedb_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    interval: dt.timedelta | None = None
    if lancedb_settings.read_consistency_seconds is not None:
        interval = dt.timedelta(seconds=lancedb_settings.read_consistency_seconds)

    # Bound the index cache so its readers (each one holds the FDs of
    # an opened ``_indices/<uuid>/...`` directory) get LRU-evicted
    # rather than leaking. Without this, a long-running daemon's FD
    # count grows monotonically until ``EMFILE``. The metadata cache
    # is intentionally left at the lancedb default (unbounded): it
    # holds parsed in-memory manifests with zero FD pressure, and a
    # cap there would just thrash. See ``LanceDBSettings`` for the
    # measurement that picked the default size.
    session = lancedb.Session(
        index_cache_size_bytes=lancedb_settings.index_cache_size_bytes,
        metadata_cache_size_bytes=None,
    )

    return await lancedb.connect_async(
        str(lancedb_dir),
        read_consistency_interval=interval,
        session=session,
    )
