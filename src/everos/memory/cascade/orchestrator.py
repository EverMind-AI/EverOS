"""Cascade orchestrator — wires watcher + scanner + worker for the lifespan.

One :class:`CascadeOrchestrator` per process. The lifespan provider
constructs it at startup, calls :meth:`start` once, and calls
:meth:`stop` at shutdown. CLI ``cascade sync`` constructs its own
instance but only invokes :meth:`drain_once` (no background tasks).

Construction is dependency-injected: the embedding / tokenizer
providers and the memory-root come in as constructor args so tests
can swap them without monkey-patching module-level singletons.
"""

from __future__ import annotations

import asyncio
import dataclasses

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import QueueSummary, md_change_state_repo

from .handlers import HandlerDeps
from .registry import build_handlers
from .scanner import CascadeScanner
from .watcher import CascadeWatcher
from .worker import CascadeWorker

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class CascadeConfig:
    """Construction-time knobs for the orchestrator.

    Defaults are sized for a lightweight (single-user / small-team) dev
    box; production tuning can surface these into
    :class:`everos.config.Settings` once the daemon has wall-clock data.
    """

    scan_interval_seconds: float = 30.0
    worker_batch_size: int = 50
    worker_max_retry: int = 3
    worker_poll_interval_seconds: float = 1.0
    worker_retry_backoff_seconds: float = 2.0


class CascadeOrchestrator:
    """Composite owner of the cascade subsystem."""

    def __init__(
        self,
        *,
        memory_root: MemoryRoot,
        embedder: EmbeddingProvider,
        tokenizer: Tokenizer,
        config: CascadeConfig | None = None,
    ) -> None:
        self._memory_root = memory_root
        self._config = config or CascadeConfig()
        deps = HandlerDeps(
            memory_root=memory_root,
            embedder=embedder,
            tokenizer=tokenizer,
        )
        self._handlers = build_handlers(deps)
        self._scanner = CascadeScanner(
            memory_root,
            scan_interval_seconds=self._config.scan_interval_seconds,
        )
        self._worker = CascadeWorker(
            self._handlers,
            batch_size=self._config.worker_batch_size,
            max_retry=self._config.worker_max_retry,
            poll_interval_seconds=self._config.worker_poll_interval_seconds,
            retry_backoff_seconds=self._config.worker_retry_backoff_seconds,
        )
        self._watcher: CascadeWatcher | None = None
        self._started = False

    async def start(self) -> None:
        """Launch the watcher (sync thread) + scanner + worker tasks.

        Before launching, reset any stale ``processing`` rows back to
        ``pending``: cascade runs single-process today, so anything in
        ``processing`` at boot is leftover from a prior crash that
        ``claim_pending_batch`` can't re-claim on its own (the WHERE
        filter is ``status='pending'``).
        """
        if self._started:
            return
        orphans = await md_change_state_repo.recover_orphan_processing()
        if orphans:
            logger.info("cascade_recovered_orphan_processing", count=orphans)
        loop = asyncio.get_running_loop()
        self._watcher = CascadeWatcher(self._memory_root, loop)
        self._watcher.start()
        await self._scanner.start()
        await self._worker.start()
        self._started = True
        logger.info("cascade_orchestrator_started")

    async def stop(self) -> None:
        """Shut everything down in reverse order."""
        if not self._started:
            return
        await self._worker.stop()
        await self._scanner.stop()
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        self._started = False
        logger.info("cascade_orchestrator_stopped")

    async def sync_once(self) -> int:
        """One scan + drain cycle (used by CLI ``cascade sync``).

        Returns the number of rows processed in this drain. The CLI
        loops on the returned count to know when to stop.
        """
        await self._scanner.scan_once()
        return await self._worker.drain_until_empty()

    async def drain_once(self) -> int:
        """Drain the queue exactly once without scanning first."""
        return await self._worker.drain_until_empty()

    async def queue_summary(self) -> QueueSummary:
        """Forward to the repo so callers don't reach past this class."""
        return await md_change_state_repo.queue_summary()
