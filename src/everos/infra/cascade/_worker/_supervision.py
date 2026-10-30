"""Supervision for the worker's long-lived background loops.

The drain, heartbeat and rebuild loops are plain ``create_task`` coroutines;
an uncaught exception would end one permanently and silently.
:class:`LoopSupervisor` runs each under a bounded restart ladder
(:data:`LOOP_RESTART_BACKOFF_SECONDS`, reset after a stable run) and, once the
ladder is spent, hands the loop's name to ``on_unrecoverable`` — by default
:func:`request_process_exit`, so a restarting process supervisor takes over.
"""

from __future__ import annotations

import asyncio
import functools
import signal
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from everos.core.observability.logging import get_logger

from ._stop import wait_or_stop
from ._tuning import LOOP_STABLE_RUN_SECONDS

logger = get_logger(__name__)


class LoopSupervisor:
    """Spawns background loops and restarts them when they raise.

    Args:
        stop: The worker's shutdown event; a loop that returns or a backoff
            interrupted by it ends supervision cleanly.
        restart_backoffs: Delay before each restart of one incident; its
            length is the restart budget.
        on_unrecoverable: Called with the loop's name once the budget is
            spent.
    """

    def __init__(
        self,
        stop: asyncio.Event,
        *,
        restart_backoffs: Sequence[float],
        on_unrecoverable: Callable[[str], None],
    ) -> None:
        self._stop = stop
        self._loop_restart_backoffs = tuple(restart_backoffs)
        self._on_unrecoverable = on_unrecoverable

    def spawn(
        self,
        loop_name: str,
        body: Callable[[], Coroutine[Any, Any, None]],
        task_name: str,
    ) -> asyncio.Task[None]:
        """Start one supervised background loop.

        Two layers, because they fail differently: :meth:`supervise` restarts
        the loop body when it raises, and the done-callback is the last-resort
        observer for the case the supervisor itself ends unexpectedly (a
        ``BaseException`` it deliberately does not catch). Without the callback
        that ending is invisible — see :data:`LOOP_RESTART_BACKOFF_SECONDS`.
        """
        task = asyncio.create_task(self.supervise(loop_name, body), name=task_name)
        task.add_done_callback(functools.partial(self._on_task_done, loop_name))
        return task

    async def supervise(
        self,
        loop_name: str,
        body: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Run ``body`` and restart it on failure, bounded, then give up.

        A clean return means the loop observed ``self._stop`` — nothing to do.
        ``CancelledError`` is re-raised so :meth:`CascadeWorker.stop` still
        works. Everything else is logged with the loop name and retried per
        :data:`LOOP_RESTART_BACKOFF_SECONDS`; when those are exhausted the
        process is asked to exit rather than run on with a dead loop.

        The budget counts **consecutive quick crashes**, not crashes over the
        process lifetime: a body that ran at least
        :data:`LOOP_STABLE_RUN_SECONDS` before raising starts a fresh
        incident. A deterministic crash-on-entry still exhausts the budget in
        ~65s; independent transients days apart each get the full ladder.
        """
        budget = len(self._loop_restart_backoffs)
        strikes = 0
        while True:
            if strikes and await wait_or_stop(
                self._stop, self._loop_restart_backoffs[strikes - 1]
            ):
                return
            if self._stop.is_set():
                return
            started = time.monotonic()
            try:
                await body()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ran = time.monotonic() - started
                if ran >= LOOP_STABLE_RUN_SECONDS:
                    strikes = 0
                strikes += 1
                logger.exception(
                    "cascade_loop_crashed",
                    loop=loop_name,
                    strike=strikes,
                    restarts_left=max(0, budget - strikes + 1),
                    ran_seconds=round(ran, 1),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if strikes > budget:
                    break
        logger.error("cascade_loop_unrecoverable", loop=loop_name, restarts=budget)
        self._on_unrecoverable(loop_name)

    def _on_task_done(self, loop_name: str, task: asyncio.Task[None]) -> None:
        """Log a supervised loop task that ended without ``stop()`` asking it to."""
        if task.cancelled() or self._stop.is_set():
            return
        exc = task.exception()
        logger.error(
            "cascade_loop_task_ended_unexpectedly",
            loop=loop_name,
            error=f"{type(exc).__name__}: {exc}" if exc is not None else None,
        )


def request_process_exit(loop_name: str) -> None:
    """Ask this process to terminate so a supervisor can restart it.

    ``SIGTERM`` to our own pid rather than ``os._exit`` so the ASGI server
    runs its graceful-shutdown path (lifespan shutdown, optimizer flush)
    instead of dropping in-flight state on the floor.

    This assumes the deployment runs under something that restarts it —
    systemd ``Restart=always``, Docker ``restart: unless-stopped``, a k8s
    Deployment. Without one the process just stops, which is still the
    better outcome: a live server whose projection pipeline is dead answers
    searches from a silently frozen index.

    The default ``on_unrecoverable``; tests pass their own rather than
    signal the pytest process.
    """
    logger.error("cascade_worker_requesting_process_exit", loop=loop_name)
    signal.raise_signal(signal.SIGTERM)
