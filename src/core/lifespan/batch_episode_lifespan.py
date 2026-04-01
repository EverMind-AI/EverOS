"""
Lifespan provider for BatchEpisodeWorker
=========================================
When ``REALTIME_EVENT_LOG_ONLY=true`` is set, this provider starts the
:class:`~biz_layer.mem_batch_episode_worker.BatchEpisodeWorker` as a background
asyncio Task during application startup and cancels it cleanly on shutdown.

The provider is registered at order 200 so it starts *after* all database /
search-engine lifespans (typically order < 100) have completed their startup.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from fastapi import FastAPI

from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.observation.logger import get_logger

logger = get_logger(__name__)


@component(name="batch_episode_lifespan_provider")
class BatchEpisodeLifespanProvider(LifespanProvider):
    """Manages the lifecycle of the BatchEpisodeWorker background task."""

    def __init__(self, order: int = 200) -> None:
        super().__init__("batch_episode_worker", order)
        self._worker = None
        self._task: Optional[asyncio.Task] = None

    async def startup(self, app: FastAPI) -> Any:
        """Start BatchEpisodeWorker if REALTIME_EVENT_LOG_ONLY=true."""
        if os.getenv("REALTIME_EVENT_LOG_ONLY", "false").lower() != "true":
            logger.info(
                "[BatchEpisode] REALTIME_EVENT_LOG_ONLY is not set – worker skipped"
            )
            return None

        from biz_layer.mem_batch_episode_worker import BatchEpisodeWorker

        self._worker = BatchEpisodeWorker()
        self._task = asyncio.create_task(self._worker.start_periodic())

        # Expose the task on app.state for external inspection / testing
        app.state.batch_episode_task = self._task

        logger.info("✅ BatchEpisodeWorker started")
        return self._task

    async def shutdown(self, app: FastAPI) -> None:
        """Stop the background task gracefully."""
        if self._worker is not None:
            self._worker.stop()

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if hasattr(app.state, "batch_episode_task"):
            delattr(app.state, "batch_episode_task")

        logger.info("[BatchEpisode] Worker stopped")
