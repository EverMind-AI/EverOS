"""The strength writer: reinforces what reads used, after they have answered.

The forgetting curve's one write (docs/configuration.md, ``[search]``). A
search hands it the decider's own core and a ``get`` the records its page
returned, as :class:`~everos.memory.ports.Reinforcement` events
(:class:`~everos.memory.ports.ReinforcementSink`); it queues them and returns
at once. One drain at a time writes everything queued: one store read for
the rows, :func:`~everos.memory.search.strengthen` per use in time order, one
transaction for the result.

Invariants:

* never on a read's path: a read appends to a bounded queue and never awaits
  a write (a full queue drops the overflow and logs it);
* only while started: the server starts it with the memory service, and a
  process that never starts it (a CLI command) queues nothing;
* the server's only writer of ``memory_strength`` (one server per memory
  root), so the read-modify-write of a drain loses no use;
* built only when ``[search] forgetting_curve`` and ``reinforce`` are on and
  the server writes derived state (``[cascade] enabled``): a read-only
  server -- an evaluation's -- has none and never reinforces.
"""

from __future__ import annotations

__all__ = ["StrengthWriter"]

import asyncio
import contextlib
from collections.abc import Sequence

from everos.core.observability.logging import get_logger
from everos.memory.ports import Reinforcement, Strength, StrengthKey, StrengthStore
from everos.memory.search import Curve, strengthen

logger = get_logger(__name__)

_MAX_PENDING = 10_000
"""Uses queued at most; more are dropped (a soft signal, never a read's wait)."""
_DRAIN_SECONDS = 10.0
"""How long a shutdown waits for what is queued to be written."""


class StrengthWriter:
    """Queues the uses reads hand it and writes them in batches.

    Args:
        store: The stored strengths.
        curve: The curve's parameters (``[search]``).
        max_pending: Uses queued at most.
        drain_seconds: How long :meth:`aclose` waits for the queue to drain.
    """

    def __init__(
        self,
        store: StrengthStore,
        curve: Curve,
        *,
        max_pending: int = _MAX_PENDING,
        drain_seconds: float = _DRAIN_SECONDS,
    ) -> None:
        self._store = store
        self._drain_seconds = drain_seconds
        self._curve = curve
        self._max_pending = max_pending
        self._pending: list[Reinforcement] = []
        self._accepting = False
        self._task: asyncio.Task[None] | None = None

    def reinforce(self, events: Sequence[Reinforcement]) -> None:
        """Queue ``events`` and make sure a drain runs; never blocks or raises."""
        if not self._accepting or not events:
            return
        room = self._max_pending - len(self._pending)
        if room < len(events):
            logger.warning(
                "strength_reinforcements_dropped",
                dropped=len(events) - max(room, 0),
                pending=len(self._pending),
            )
        if room <= 0:
            return
        self._pending.extend(events[:room])
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain(), name="strength-writer")

    async def start(self) -> None:
        """Accept uses from now on."""
        self._accepting = True

    async def flush(self) -> None:
        """Wait until everything queued so far is written."""
        while self._task is not None and not self._task.done():
            await asyncio.shield(self._task)

    async def aclose(self) -> None:
        """Stop accepting and write what is queued (waiting a bounded time)."""
        self._accepting = False
        try:
            async with asyncio.timeout(self._drain_seconds):
                await self.flush()
        except TimeoutError:
            logger.warning("strength_writer_drain_timeout", pending=len(self._pending))
            if self._task is not None:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

    async def _drain(self) -> None:
        while self._pending:
            batch, self._pending = self._pending, []
            try:
                await self._write(batch)
            except Exception:  # a soft signal: a failed batch never breaks a read
                logger.warning(
                    "strength_write_failed", reinforcements=len(batch), exc_info=True
                )

    async def _write(self, batch: Sequence[Reinforcement]) -> None:
        """One batch: read its rows, apply its uses in time order, write once."""
        stored = await self._store.read({event.key for event in batch})
        changed: dict[StrengthKey, Strength] = {}
        for event in sorted(batch, key=lambda e: e.at):
            prior = changed.get(event.key) or stored.get(event.key)
            changed[event.key] = strengthen(self._curve, prior, event)
        await self._store.write(changed)
        logger.debug(
            "strength_reinforced", memories=len(changed), reinforcements=len(batch)
        )
