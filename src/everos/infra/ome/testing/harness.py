"""StrategyTestHarness — full OfflineEngine on a tmp SQLite db.

Designed for end-to-end strategy tests: register, start, emit, drain
until terminal, inspect run records. Cleans up the tmp directory on exit.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.records import RunRecord, RunStatus


class StrategyTestHarness:
    """Async context manager wrapping OfflineEngine on a tmp SQLite db.

    Provides a test-friendly interface to register strategies, emit events,
    and inspect run records.

    Example:
        async with StrategyTestHarness() as h:
            h.register(my_strategy_func)
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
            shutil.rmtree(self._tmpdir, ignore_errors=True)  # noqa: SLF001

    def register(self, func: Any) -> None:
        """Register a strategy function.

        Args:
            func: A function decorated with @offline_strategy.
        """
        self._engine.register(func)

    async def start(self) -> None:
        """Start the OfflineEngine."""
        await self._engine.start()

    async def emit(self, event: BaseEvent) -> None:
        """Emit an event to the engine.

        Args:
            event: A BaseEvent subclass instance.
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
                f"{self._engine._active_runs} in-flight runs after {timeout}s"  # noqa: SLF001
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
        return await self._engine._run_record_store.list_runs(  # noqa: SLF001
            strategy_name=strategy_name,
            status=status,
        )
