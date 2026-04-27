"""
LongJob Runner - Used to start and manage long-running tasks

Provides functionality to run a single long-running task, including:
- Finding the specified long job via DI
- Graceful startup and shutdown
- Handling shutdown based on asyncio task cancellation mechanism
- Error handling and logging
"""

import asyncio
from typing import Optional

from core.di.utils import get_bean
from core.longjob.interfaces import LongJobInterface
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def run_longjob_mode(longjob_name: str):
    """
    Run the specified long-running job mode

    This function runs as an asyncio Task and is triggered to shut down via task.cancel().
    When CancelledError is received, the long-running job will be gracefully shut down.

    Args:
        longjob_name: Name of the long-running job
    """
    logger.info("🚀 Starting LongJob mode: %s", longjob_name)

    longjob_instance: Optional[LongJobInterface] = None

    try:
        # Try to get the specified long-running job from the DI container
        try:
            longjob_instance = get_bean(longjob_name)
            logger.info(
                "✅ Found long-running job: %s (%s)",
                longjob_name,
                type(longjob_instance).__name__,
            )
        except Exception as e:
            logger.error(
                "❌ Unable to find long-running job '%s': %s", longjob_name, str(e)
            )
            logger.info(
                "💡 Please ensure the long-running job is correctly registered in the DI container"
            )
            return

        # Check if it is an implementation of LongJobInterface
        if not isinstance(longjob_instance, LongJobInterface):
            logger.error(
                "❌ '%s' is not an implementation of LongJobInterface", longjob_name
            )
            logger.info(
                "💡 Long-running jobs must inherit from LongJobInterface or its subclasses"
            )
            return

        # Start the long-running job
        logger.info("🔄 Starting long-running job: %s", longjob_name)
        await longjob_instance.start()

        logger.info(
            "✅ Long-running job '%s' has started and is running...", longjob_name
        )

        # Wait indefinitely until the task is canceled
        # Use an uncompleted Future to keep the task running
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        # Received task cancel signal, begin graceful shutdown
        logger.info(
            "🛑 Received cancellation signal, starting graceful shutdown of long-running job: %s",
            longjob_name,
        )
        if longjob_instance:
            try:
                await longjob_instance.shutdown()
                logger.info(
                    "✅ Long-running job '%s' has been successfully shut down",
                    longjob_name,
                )
            except Exception as e:
                logger.error(
                    "❌ Error during long-running job shutdown: %s",
                    str(e),
                    exc_info=True,
                )
        # Re-raise CancelledError so the caller knows the task was cancelled
        raise

    except Exception as e:
        # Exception occurred during execution
        logger.error("❌ Error running long-running job: %s", str(e), exc_info=True)
        if longjob_instance:
            try:
                await longjob_instance.shutdown()
                logger.info("✅ Long-running job has been shut down after exception")
            except Exception as shutdown_error:
                logger.error(
                    "❌ Error during long-running job shutdown: %s",
                    str(shutdown_error),
                    exc_info=True,
                )
