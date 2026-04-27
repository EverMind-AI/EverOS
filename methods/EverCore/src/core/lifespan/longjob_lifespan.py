"""
LongJob lifecycle provider implementation

Used to manage the lifecycle of long-running tasks, including startup and shutdown.
"""

import asyncio
from fastapi import FastAPI
from typing import Optional, Any
import os
from core.observation.logger import get_logger
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider

logger = get_logger(__name__)


@component(name="longjob_lifespan_provider")
class LongJobLifespanProvider(LifespanProvider):
    """LongJob lifecycle provider"""

    def __init__(self, name: str = "longjob", order: int = 100):
        """
        Initialize LongJob lifecycle provider

        Args:
            name (str): Provider name
            order (int): Execution order; LongJob should start after all infrastructure is up
        """
        super().__init__(name, order)
        self._longjob_task: Optional[asyncio.Task] = None
        self._longjob_name: Optional[str] = None

    async def startup(self, app: FastAPI) -> Any:
        """
        Start LongJob task

        Core logic: asyncio.create_task(run_longjob_mode(longjob_name))

        Args:
            app (FastAPI): FastAPI application instance

        Returns:
            Any: Reference to the LongJob task
        """
        try:
            from core.longjob.longjob_runner import run_longjob_mode

            self._longjob_name = os.getenv("LONGJOB_NAME")
            if not self._longjob_name:
                logger.warning(
                    "⚠️ LONGJOB_NAME environment variable not set, skipping LongJob startup"
                )
                return None
            # Core logic: create an async task to run the long-running job
            self._longjob_task = asyncio.create_task(
                run_longjob_mode(self._longjob_name)
            )

            # Store the task in app.state for access elsewhere
            app.state.longjob_task = self._longjob_task

            logger.info("✅ LongJob task started: %s", self._longjob_name)

            return self._longjob_task

        except Exception as e:
            logger.error("❌ Error starting LongJob: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        Shut down LongJob task

        Args:
            app (FastAPI): FastAPI application instance
        """
        if not self._longjob_task:
            logger.info("No running LongJob task")
            return

        logger.info("Shutting down LongJob: %s", self._longjob_name)

        try:
            # Cancel the task
            if not self._longjob_task.done():
                self._longjob_task.cancel()
                try:
                    await self._longjob_task
                except asyncio.CancelledError:
                    logger.info("✅ LongJob task cancelled: %s", self._longjob_name)
            else:
                logger.info("✅ LongJob task completed: %s", self._longjob_name)

        except Exception as e:
            logger.error("❌ Error shutting down LongJob: %s", str(e))

        # Clean up LongJob-related attributes in app.state
        if hasattr(app.state, 'longjob_task'):
            delattr(app.state, 'longjob_task')
