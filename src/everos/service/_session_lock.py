"""Per-session asyncio lock for serialising concurrent memorize() calls.

Two concurrent ``POST /add`` (or ``/flush``) calls on the **same**
``session_id`` race on the unprocessed_buffer:

1. Both read ``unprocessed_buffer`` for the session and see the same
   pre-existing rows.
2. Both run boundary detection independently against their own merged
   slice (each task only sees its own newly-arrived messages plus the
   shared pre-existing buffer rows — neither sees the other's messages).
3. Both call ``_replace_buffer(session_id, tail)`` — the later write
   silently overwrites the earlier write's tail and the earlier task's
   tail messages are lost forever (they never made it into any memcell
   either, since each task's boundary call only saw its own slice).

This module serialises memorize() at the ``session_id`` granularity so
the read-merge-boundary-write cycle is atomic per session.

Cross-process safety is out of scope (single-instance everos; would
need fcntl on the sqlite db). Cross-session calls remain fully parallel.

Wrap acquire + work in ``asyncio.timeout(...)`` (see
``MemorizeSettings.session_lock_timeout_seconds``) so a hung LLM cannot
hold the lock forever — on timeout the task is cancelled and
``async with`` releases the lock automatically.
"""

from __future__ import annotations

import asyncio

# Plain dict (not WeakValueDictionary): a Lock with pending waiters must
# outlive the dict entry, otherwise GC racing with waiters can drop a
# lock mid-flight (CPython bpo-28427). Same rationale as
# ``everos.core.persistence.markdown.writer.MarkdownWriter._path_locks``.
_session_locks: dict[str, asyncio.Lock] = {}


def get_session_lock(session_id: str) -> asyncio.Lock:
    """Return the per-session asyncio Lock; create on first use.

    ``dict.setdefault`` is atomic under single-threaded asyncio (no GIL
    release between the get and the insert), so no meta-lock is needed
    around the registry.
    """
    return _session_locks.setdefault(session_id, asyncio.Lock())


def _reset_for_tests() -> None:
    """Test-only: drop all registered locks.

    Used by integration test fixtures that rebuild memorize singletons
    against a fresh tmp memory_root; ensures no stale lock state leaks
    across tests.
    """
    _session_locks.clear()
