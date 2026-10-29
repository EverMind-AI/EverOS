"""Per-session asyncio locks that serialise concurrent memorize() calls.

Two concurrent ``POST /add`` (or ``/flush``) calls on the **same** session
race on the unprocessed buffer:

1. Both read the session's buffer and see the same pre-existing rows.
2. Both run boundary detection against their own merged slice (each sees
   only its own new messages plus the shared buffer rows).
3. Both replace the buffer with their tail -- the later write silently
   overwrites the earlier one, and the earlier call's tail messages are lost
   (they never reached a memcell either).

Holding one lock per ``(app_id, project_id, session_id)`` across the
read-merge-boundary-write cycle makes it atomic per session; different
sessions stay fully parallel. Cross-process safety is out of scope (one
EverOS process owns a memory root). The caller bounds the wait with
``asyncio.timeout`` (``[memorize] session_lock_timeout_seconds``).
"""

from __future__ import annotations

__all__ = ["SessionLocks"]

import asyncio


class SessionLocks:
    """The memorize lock of every session one :class:`MemoryService` has seen.

    A plain dict, not a ``WeakValueDictionary``: a lock with pending waiters
    must outlive its entry, or GC racing the waiters can drop a lock
    mid-flight (CPython bpo-28427). Same rationale as the markdown writer's
    path locks.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> asyncio.Lock:
        """The lock for ``key``, created on first use.

        ``dict.setdefault`` is atomic under single-threaded asyncio, so the
        registry needs no lock of its own.
        """
        return self._locks.setdefault(key, asyncio.Lock())
