"""Per-strategy partition locks for serialising RMW critical sections.

The OME engine intentionally does NOT serialise concurrent runs of the
same strategy
(``local/specs/2026-04-27-ome-tech-design.md`` §4.5.2: the business logic
must guard itself, deciding inside the strategy body via
``async with lock`` bucketed by a business key). Offline strategies whose
body is a read →
modify → write on shared state (cluster rows, user.md, SKILL.md)
serialise on a business key (``owner_id`` / ``agent_id``) here.

The locks live in :data:`everos.core.locks.LOCKS` under the kind
``partition:<strategy_name>``: one ``asyncio.Lock`` per
``(strategy_name, partition_key)`` pair, never evicted (see that module).
The table grows with the live partition-key set, which in practice is
bounded by the agent / user / cluster counts a single everos process owns.

No separate acquire timeout is imposed here. OME's existing attempt timeout
still includes partition waiting and reacquiring execution capacity; a timed
out attempt follows the normal retry / dead-letter policy. Direct callers
without an executor wait until the lock is available or they are cancelled.

Cross-process safety is out of scope: everos is single-process by
design (see ``CLAUDE.md`` deployment notes); the enterprise edition
layers a distributed coordinator on top.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from everos.core.concurrency import acquire_without_slot
from everos.core.locks import LOCKS


def get_partition_lock(
    strategy_name: str, partition_key: str
) -> AbstractAsyncContextManager[None]:
    """Hold the lock for ``(strategy_name, partition_key)`` with ``async with``.

    Every call returns a fresh context manager over the same lock; waiters
    on one key acquire it in FIFO order. An OME attempt returns its execution
    slot while waiting and reclaims it before entering the body. Direct calls
    outside an executor keep ordinary lock semantics.
    """
    return acquire_without_slot(LOCKS.hold(f"partition:{strategy_name}", partition_key))


def user_cluster_lock(
    app_id: str, project_id: str, owner_id: str
) -> AbstractAsyncContextManager[None]:
    """The lock every writer of one user's clusters holds: profile clustering,
    Reflection folding members in, and deletion taking them out."""
    return get_partition_lock(
        "trigger_profile_clustering", f"{app_id}:{project_id}:{owner_id}"
    )


def _reset_for_tests() -> None:
    """Test-only: drop every lock (see :meth:`everos.core.locks.NamedLocks.reset`)."""
    LOCKS.reset()
