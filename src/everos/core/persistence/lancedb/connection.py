"""Async LanceDB connection factory.

LanceDB does not live inside the SQLAlchemy ecosystem; it has its own
``connect_async`` returning :class:`lancedb.AsyncConnection`. This module
is a thin wrapper that:

    1. ensures the lancedb root directory exists
    2. converts ``LanceDBSettings.read_consistency_seconds`` into the
       :class:`datetime.timedelta` value LanceDB expects
    3. installs a :class:`lancedb.Session` with **both** caches capped, so
       index readers cannot pin file descriptors for the life of the process
       (see :attr:`LanceDBSettings.metadata_cache_size_bytes` for the
       measurement that shows the metadata cap is the decisive one)
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
            ``index_cache_size_bytes`` / ``metadata_cache_size_bytes`` cap
            the session's two caches.

    Returns:
        An :class:`AsyncConnection` ready for table operations.
    """
    # mkdir is a microsecond-fast syscall and only fires on first connect;
    # not worth pulling in anyio.Path / aiofiles for it.
    lancedb_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    interval: dt.timedelta | None = None
    if lancedb_settings.read_consistency_seconds is not None:
        interval = dt.timedelta(seconds=lancedb_settings.read_consistency_seconds)

    # Both caches must be bounded. Each cached index reader holds the FDs of
    # an opened ``_indices/<uuid>/...`` directory, and POSIX does not release
    # those on unlink, so an entry that outlives its files pins descriptors
    # for the life of the process. Lance keys metadata entries by dataset
    # version and does not evict them when cleanup deletes the files
    # (lance-format/lance#7718), which is why leaving *that* cache unbounded
    # — as this module used to — is what made the count grow without end.
    # See ``LanceDBSettings`` for the measurement behind both sizes.
    session = lancedb.Session(
        index_cache_size_bytes=lancedb_settings.index_cache_size_bytes,
        metadata_cache_size_bytes=lancedb_settings.metadata_cache_size_bytes,
    )

    return await lancedb.connect_async(
        str(lancedb_dir),
        read_consistency_interval=interval,
        session=session,
    )
