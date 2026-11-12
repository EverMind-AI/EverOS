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
from typing import TYPE_CHECKING

from everos.core.locks import LOCKS

if TYPE_CHECKING:
    from everos.memory.ports import Scope


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


def profile_lock(
    scope: Scope, person_id: str | None = None
) -> AbstractAsyncContextManager[None]:
    """The lock every writer of one person's profile in one space holds.

    The space is ``scope``: a user's own (``users/<u>/user.md``, the person
    is the user) or a group (``groups/<g>/members/<m>.md``, the person is the
    member ``person_id`` -- docs/group-memory.md, section 7.4). The profile
    update (a read, a model call, a write), the picture route's poster-only
    items, and a forget, its undo or a person's edit taking items out and
    putting them back all hold it: none of them writes back a profile another
    changed meanwhile. Different people, and the members of one segment,
    update in parallel.

    Raises:
        ValueError: A group scope without the member.
    """
    if scope.owner_type == "group" and person_id is None:
        raise ValueError("a profile in a group belongs to one member")
    return get_partition_lock("profile", _key(scope, person_id or scope.owner_id))


def scenes_lock(scope: Scope) -> AbstractAsyncContextManager[None]:
    """The lock every writer of one owner's scenes holds (docs/scenes.md).

    A user's or a group's: the scene strategy places an episode and writes a
    version under it; a forget, its undo or an update that reaches the
    owner's scenes changes them under it.
    """
    return get_partition_lock("scenes", _key(scope))


def forget_lock(scope: Scope) -> AbstractAsyncContextManager[None]:
    """One forget, undo or update of one owner's memory at a time.

    A user's or a group's (docs/group-memory.md, section 9.4): a forget plans
    from what the owner holds and changes it, and an update checks the
    version it replaces is current; two at once could each act on what the
    other is changing. Taken before any other lock they hold (the scene,
    profile and owner locks).
    """
    return get_partition_lock("forget", _key(scope))


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


def _key(scope: Scope, *parts: str) -> str:
    """One owner's partition key: its type first, as user and group ids share
    one id space (docs/group-memory.md, section 4.1)."""
    return ":".join(
        (scope.owner_type, scope.app_id, scope.project_id, scope.owner_id, *parts)
    )


def _reset_for_tests() -> None:
    """Test-only: drop every lock (see :meth:`everos.core.locks.NamedLocks.reset`)."""
    LOCKS.reset()
