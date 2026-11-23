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
from everos.core.observability.logging import get_logger
from everos.core.utils.open_files import open_file_usage

logger = get_logger(__name__)

INDEX_CACHE_BYTES = 16 * 1024 * 1024
"""Index cache when the open-file soft limit is low or unknown: the cap
measured to hold descriptors near 290 (``LanceDBSettings``)."""
LARGE_INDEX_CACHE_BYTES = 64 * 1024 * 1024
"""Index cache when the soft limit is at least
:data:`LARGE_INDEX_CACHE_MIN_SOFT_LIMIT`. Holds a 161k-row table's IVF_RQ
index (24 MB) with its BM25 readers, so hybrid search stops re-reading them
(three lanes 30.5 -> 20.3 ms; 128 MB was no faster). 200 rounds of writes,
optimize and retrains kept descriptors flat (max 75 against 65 at 16 MB;
unbounded climbed to 696), 2026-11. Never larger: the cap is what bounds the
stale readers LanceDB keeps after optimize."""
LARGE_INDEX_CACHE_MIN_SOFT_LIMIT = 8192
"""``everos server start`` raises the soft limit to 32768 where the hard
limit allows; a host left at 1024 (direct ``uvicorn``, a container's hard
limit) keeps the 16 MB cap."""


def index_cache_bytes(configured: int | None, usage: tuple[int, int] | None) -> int:
    """The index cache cap: ``configured`` when set, else chosen from the
    open-file soft limit in ``usage`` (``(open, soft)``; ``None`` where it
    cannot be read, e.g. Windows)."""
    if configured is not None:
        return configured
    if usage is not None and usage[1] >= LARGE_INDEX_CACHE_MIN_SOFT_LIMIT:
        return LARGE_INDEX_CACHE_BYTES
    return INDEX_CACHE_BYTES


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
            ``index_cache_size_bytes`` (or :func:`index_cache_bytes`'s choice
            when unset) / ``metadata_cache_size_bytes`` cap the session's two
            caches.

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
    usage = open_file_usage()
    index_cache = index_cache_bytes(lancedb_settings.index_cache_size_bytes, usage)
    session = lancedb.Session(
        index_cache_size_bytes=index_cache,
        metadata_cache_size_bytes=lancedb_settings.metadata_cache_size_bytes,
    )
    logger.info(
        "lancedb_session_caches",
        index_cache_bytes=index_cache,
        metadata_cache_bytes=lancedb_settings.metadata_cache_size_bytes,
        open_file_soft_limit=usage[1] if usage else None,
    )

    return await lancedb.connect_async(
        str(lancedb_dir),
        read_consistency_interval=interval,
        session=session,
    )
