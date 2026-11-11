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

No acquire timeout: an OME strategy run has no upstream client
waiting on it, so timing out a queued caller would only convert
"slow" into a permanent ``dead_letter`` data-loss (`max_retries`
exhaustion). The LLM client owns the per-request timeout
(`component.llm.openai_provider`, default 60s) — that is the layer
that breaks a stuck LLM call, not this one. If a genuinely hung
strategy holds the lock indefinitely it surfaces as a stuck queue
under process-level monitoring; the recovery is a process restart,
not a silent data drop.

Cross-process safety is out of scope: everos is single-process by
design (see ``CLAUDE.md`` deployment notes); the enterprise edition
layers a distributed coordinator on top.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from everos.core.locks import LOCKS


def get_partition_lock(
    strategy_name: str, partition_key: str
) -> AbstractAsyncContextManager[None]:
    """Hold the lock for ``(strategy_name, partition_key)`` with ``async with``.

    Every call returns a fresh context manager over the same lock; waiters
    on one key acquire it in FIFO order.
    """
    return LOCKS.hold(f"partition:{strategy_name}", partition_key)


def user_cluster_lock(
    app_id: str, project_id: str, owner_id: str
) -> AbstractAsyncContextManager[None]:
    """The lock every writer of one user's clusters holds: profile clustering,
    Reflection folding members in, and deletion taking them out."""
    return get_partition_lock(
        "trigger_profile_clustering", f"{app_id}:{project_id}:{owner_id}"
    )


def member_profile_lock(
    app_id: str, project_id: str, group_id: str, member_id: str
) -> AbstractAsyncContextManager[None]:
    """The lock every writer of one member's profile in one group holds.

    One per ``(group, member)`` (docs/group-memory.md, section 7.4): two
    updates of one member's profile never lose each other's items, and
    updates of different members run in parallel. The picture route's
    poster-only items and the ``update_member_profiles`` strategy share it.
    """
    return get_partition_lock(
        "update_member_profiles", f"{app_id}:{project_id}:{group_id}:{member_id}"
    )


def group_profile_lock(
    app_id: str, project_id: str, group_id: str
) -> AbstractAsyncContextManager[None]:
    """The lock every writer of one group's profile batch holds.

    One per group (docs/group-memory.md, section 7.5): a segment queued while
    an update runs waits for it, so no segment is read twice or dropped from
    the batch, and the update reads the items the previous one wrote.
    """
    return get_partition_lock(
        "update_group_profile", f"{app_id}:{project_id}:{group_id}"
    )


def _reset_for_tests() -> None:
    """Test-only: drop every lock (see :meth:`everos.core.locks.NamedLocks.reset`)."""
    LOCKS.reset()
