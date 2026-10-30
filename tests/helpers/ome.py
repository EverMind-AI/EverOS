"""Declare throw-away OME strategies in tests the way production does.

Production code declares a :class:`~everos.memory.ports.StrategySpec` and
registers ``to_ome_strategy(spec, deps)``; :func:`ome_strategy` is that pair
as a decorator, so an engine test states a strategy in one place::

    @ome_strategy(name="s", trigger=Immediate(on=[Ping]), emits=[Pong])
    async def s(event: Ping, deps: MemoryDeps) -> list[Event]:
        return [Pong()]

    engine.register(s)

:class:`StrategyTestHarness` runs a real engine on a temporary database for
end-to-end strategy tests: register, start, emit, drain until terminal,
inspect run records.
"""

from __future__ import annotations

__all__ = ["StrategyTestHarness", "ome_strategy", "strategy_spec"]

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from everos.infra.ome import (
    OfflineEngine,
    OMEConfig,
    RunRecord,
    RunStatus,
    to_ome_strategy,
)
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, StrategyFn, StrategySpec, Trigger
from tests.fakes import make_deps


def strategy_spec(
    *,
    name: str,
    trigger: Trigger,
    emits: Iterable[type[Event]] = (),
    **options: Any,
) -> Callable[[StrategyFn], StrategySpec]:
    """Declare the decorated function as a :class:`StrategySpec`.

    ``options`` are the remaining spec fields (``applies_to``, ``gate``,
    ``max_retries``, ``enabled``).
    """

    def wrap(fn: StrategyFn) -> StrategySpec:
        return StrategySpec(
            name=name, fn=fn, trigger=trigger, emits=frozenset(emits), **options
        )

    return wrap


def ome_strategy(
    *,
    name: str,
    trigger: Trigger,
    emits: Iterable[type[Event]] = (),
    deps: MemoryDeps | None = None,
    **options: Any,
) -> Callable[[StrategyFn], Any]:
    """Declare the decorated function as a spec and adapt it for OME.

    ``deps`` defaults to in-memory doubles; the remaining keywords are as
    for :func:`strategy_spec`.
    """

    def wrap(fn: StrategyFn) -> Any:
        spec = strategy_spec(name=name, trigger=trigger, emits=emits, **options)(fn)
        return to_ome_strategy(spec, deps if deps is not None else make_deps())

    return wrap


class StrategyTestHarness:
    """Async context manager wrapping OfflineEngine on a tmp SQLite db.

    Provides a test-friendly interface to register strategies, emit events,
    and inspect run records.

    Example:
        async with StrategyTestHarness() as h:
            h.register(my_spec, deps)
            await h.start()
            await h.emit(MyEvent())
            await h.drain(timeout=5)
            runs = await h.list_runs("my_strategy")
            assert len(runs) == 1
    """

    def __init__(self) -> None:
        """Initialize a StrategyTestHarness with a temp SQLite db."""
        self._tmpdir = Path(mkdtemp(prefix="ome_test_"))
        cfg = OMEConfig(
            jobstore_path=self._tmpdir / "ome.db",
            config_watch=False,
            max_concurrent_runs=20,
            max_retries=1,
            retry_backoff_base_seconds=0.0,
        )
        self._engine = OfflineEngine(config=cfg)

    async def __aenter__(self) -> StrategyTestHarness:
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Exit the async context and clean up temp resources."""
        try:
            await self._engine.stop()
        finally:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def register(self, spec: StrategySpec, deps: MemoryDeps) -> None:
        """Register ``spec`` with ``deps`` bound, as production does.

        Args:
            spec: The strategy declaration.
            deps: What the strategy function receives on every run.
        """
        self._engine.register(to_ome_strategy(spec, deps))

    async def start(self) -> None:
        """Start the OfflineEngine."""
        await self._engine.start()

    async def emit(self, event: Event) -> None:
        """Emit an event to the engine.

        Args:
            event: A Event subclass instance.
        """
        await self._engine.emit(event)

    async def drain(self, *, timeout: float = 30.0) -> None:  # noqa: ASYNC109
        """Wait until every enqueued strategy run has finished.

        Delegates to :meth:`OfflineEngine.wait_idle`, which tracks runs
        from the moment ``_enqueue_run`` bumps the counter (so a caller
        that ``emit``s then immediately ``drain``s does NOT see false-
        idle while APS is still launching the coroutine). Polling
        ``find_running`` alone — the previous implementation — missed
        that gap between ``add_job`` and ``mark_running`` and let tests
        race past in-flight jobs.

        Args:
            timeout: Maximum seconds to wait, defaults to 30.0.

        Raises:
            TimeoutError: if runs remain in flight after ``timeout`` seconds.
        """
        if not await self._engine.wait_idle(timeout=timeout):
            raise TimeoutError(
                f"drain: engine still has "
                f"{self._engine._in_flight.active} in-flight runs after {timeout}s"
            )

    async def list_runs(
        self,
        strategy_name: str,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List run records for a strategy, optionally filtered by status.

        Args:
            strategy_name: The name of the strategy.
            status: Optional status filter (e.g. RunStatus.SUCCESS).

        Returns:
            A list of RunRecord objects.
        """
        return await self._engine._run_record_store.list_runs(
            strategy_name=strategy_name,
            status=status,
        )
