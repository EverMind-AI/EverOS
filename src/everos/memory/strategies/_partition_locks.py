"""Per-strategy partition locks for serialising RMW critical sections.

The OME engine intentionally does NOT serialise concurrent runs of the
same strategy
(``local/specs/2026-04-27-ome-tech-design.md`` §4.5.2: the business logic
must guard itself, deciding inside the strategy body via
``async with lock`` bucketed by a business key). Offline strategies whose
body is a read →
modify → write on shared state (cluster rows, user.md, SKILL.md)
serialise on a business key (``owner_id`` / ``agent_id``) here.

Mirrors :mod:`everos.service._session_lock` (and
:class:`everos.core.persistence.markdown.writer.MarkdownWriter`'s
per-path lock pool): one ``asyncio.Lock`` per
``(strategy_name, partition_key)`` pair, **never evicted** — a lock
with pending waiters must outlive any dict entry that points to it,
otherwise GC racing waiters can drop the lock mid-flight (CPython
bpo-28427). The pool grows with the live partition-key set, which in
practice is bounded by the agent / user / cluster counts a single
everos process owns.

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

import asyncio

_pools: dict[str, dict[str, asyncio.Lock]] = {}


def get_partition_lock(strategy_name: str, partition_key: str) -> asyncio.Lock:
    """Return the lock for ``(strategy_name, partition_key)``; create on first use.

    ``dict.setdefault`` is atomic under single-threaded asyncio — no
    ``await`` runs between the nested ``setdefault`` calls, so the
    "check then insert" pair is indivisible. No meta-lock is needed.

    Callers acquire the lock with ``async with``; the lock object is
    cached forever (see module docstring on bpo-28427), and the inner
    asyncio queue gives FIFO fairness across waiters on the same key.
    """
    return _pools.setdefault(strategy_name, {}).setdefault(
        partition_key, asyncio.Lock()
    )


def _reset_for_tests() -> None:
    """Test-only: drop every registered lock pool.

    Used by test fixtures that need a clean lock registry between
    cases (no inherited holders, no inherited waiters).
    """
    _pools.clear()
