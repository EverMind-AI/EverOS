"""
Recycle consumer base implementation.
Base implementation of recycle consumer.
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from datetime import datetime

from core.longjob.interfaces import (
    LongJobInterface,
    LongJobStatus,
    ConsumerConfig,
    ErrorHandler,
    MessageBatch,
)
from common_utils.datetime_utils import get_now_with_timezone


class DefaultErrorHandler(ErrorHandler):
    """Default error handler"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Default error handling: log error and return True to continue execution

        Args:
            error: The exception occurred
            context: Error context information

        Returns:
            bool: Whether to continue execution
        """
        self.logger.error(
            f"Error in consumer {context.get('job_id', 'unknown')}: {str(error)}",
            exc_info=True,
            extra=context,
        )
        return True


class RecycleConsumerBase(LongJobInterface, ABC):
    """
    Base implementation of recycle consumer
    Provides a basic framework for continuous consumption, including error handling, retry logic, timeout handling, etc.
    """

    def __init__(
        self,
        job_id: str,
        config: Optional[Dict[str, Any]] = None,
        consumer_config: Optional[ConsumerConfig] = None,
    ):
        """
        Initialize recycle consumer

        Args:
            job_id: Job ID
            config: Base configuration
            consumer_config: Consumer-specific configuration
        """
        super().__init__(job_id, config)
        self.consumer_config = consumer_config or ConsumerConfig()
        self.logger = logging.getLogger(f"{__name__}.{job_id}")
        self._task: Optional[asyncio.Task] = None
        self._error_handler = self.consumer_config.error_handler or DefaultErrorHandler(
            self.logger
        )

        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'total_timeouts': 0,
            'start_time': None,
            'last_processed_time': None,
        }

    async def start(self) -> None:
        """Start consumer"""
        if self.status in [LongJobStatus.RUNNING, LongJobStatus.STARTING]:
            self.logger.warning(
                "Consumer %s is already running or starting", self.job_id
            )
            return

        self.logger.info("Starting consumer %s", self.job_id)
        self.status = LongJobStatus.STARTING

        try:
            # Initialize resources
            await self._initialize()

            # Start consumption loop
            self._task = asyncio.create_task(self._consume_loop())
            self.status = LongJobStatus.RUNNING
            self.stats['start_time'] = get_now_with_timezone()

            self.logger.info("Consumer %s started successfully", self.job_id)

        except Exception as e:
            self.status = LongJobStatus.ERROR
            self.logger.error(
                "Failed to start consumer %s: %s", self.job_id, str(e), exc_info=True
            )
            raise

    async def shutdown(
        self, timeout: float = 30.0, wait_for_current_task: bool = True
    ) -> None:
        """
        Gracefully shutdown consumer

        Args:
            timeout: Shutdown timeout in seconds
            wait_for_current_task: Whether to wait for current task to complete
        """
        if self.status in [LongJobStatus.STOPPED, LongJobStatus.STOPPING]:
            self.logger.warning(
                "Consumer %s is already stopped or stopping", self.job_id
            )
            return

        self.logger.info("Gracefully shutting down consumer %s", self.job_id)
        self.status = LongJobStatus.STOPPING

        # Request stop
        self.request_stop()

        # Wait for task completion
        if self._task and not self._task.done():
            if wait_for_current_task:
                try:
                    # Wait for current message processing to complete
                    self.logger.info(
                        "Waiting for current task to complete in consumer %s",
                        self.job_id,
                    )
                    await asyncio.wait_for(self._task, timeout=timeout)
                    self.logger.info(
                        "Current task completed gracefully in consumer %s", self.job_id
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "Consumer %s shutdown timeout after %ss, cancelling task",
                        self.job_id,
                        timeout,
                    )
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        self.logger.info("Task cancelled in consumer %s", self.job_id)
            else:
                # Cancel task immediately
                self.logger.info(
                    "Immediately cancelling task in consumer %s", self.job_id
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        # Cleanup resources
        try:
            await self._cleanup()
        except Exception as cleanup_error:
            self.logger.error(
                "Error during cleanup: %s", str(cleanup_error), exc_info=True
            )

        self.status = LongJobStatus.STOPPED
        self.logger.info("Consumer %s shutdown completed", self.job_id)

    async def _consume_loop(self) -> None:
        """Main consumption loop"""
        self.logger.info("Consumer %s entering consume loop", self.job_id)

        while not self.should_stop():
            try:
                # Check if there are messages to consume
                if not await self._has_messages():
                    await asyncio.sleep(0.1)  # Brief sleep to avoid high CPU usage
                    continue

                # Consume messages
                await self._consume_messages()

            except Exception as e:
                # Error handling
                context = {
                    'job_id': self.job_id,
                    'timestamp': get_now_with_timezone().isoformat(),
                    'stats': self.stats.copy(),
                }

                self.stats['total_errors'] += 1

                try:
                    should_continue = await self._error_handler.handle_error(e, context)
                    if not should_continue:
                        self.logger.error(
                            "Error handler requested stop for consumer %s", self.job_id
                        )
                        break
                except Exception as handler_error:
                    self.logger.error(
                        "Error in error handler for consumer %s: %s",
                        self.job_id,
                        str(handler_error),
                        exc_info=True,
                    )
                    # If error handler itself fails, sleep briefly and continue
                    await asyncio.sleep(1.0)

        self.logger.info("Consumer %s exiting consume loop", self.job_id)

    async def _consume_messages(self) -> None:
        """Core logic for consuming messages"""
        timeout = self.consumer_config.timeout

        # Business logic handles batch processing itself; here only process single message
        if self.should_stop():
            return

        try:
            # Use timeout to control processing time for a single message
            await asyncio.wait_for(self._process_single_message(), timeout=timeout)

            self.stats['total_processed'] += 1
            self.stats['last_processed_time'] = get_now_with_timezone()

        except asyncio.TimeoutError:
            self.stats['total_timeouts'] += 1
            self.logger.warning(
                "Message processing timeout in consumer %s (timeout: %ss)",
                self.job_id,
                timeout,
            )
            # Timeout is also treated as an error, handled by error handler
            timeout_error = TimeoutError(f"Message processing timeout ({timeout}s)")
            context = {
                'job_id': self.job_id,
                'error_type': 'timeout',
                'timeout': timeout,
                'timestamp': get_now_with_timezone().isoformat(),
            }

            try:
                should_continue = await self._error_handler.handle_error(
                    timeout_error, context
                )
                if not should_continue:
                    raise timeout_error
            except Exception as handler_error:
                self.logger.error(
                    "Error in timeout error handler: %s",
                    str(handler_error),
                    exc_info=True,
                )

        except Exception as e:
            # Other exceptions will be caught and handled in the outer loop
            raise

    async def _process_single_message(self) -> None:
        """
        Process a single message with enhanced retry logic
        First fetch the message, then pass the same message batch during retries
        """
        retry_config = self.consumer_config.retry_config
        last_error = None
        message_batch = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                # Fetch message on first attempt, use same batch on retries
                if attempt == 0:
                    raw_message = await self._fetch_message()
                    if raw_message is None:
                        return  # No message to process

                    # If not MessageBatch, wrap automatically
                    if isinstance(raw_message, MessageBatch):
                        message_batch = raw_message
                    else:
                        message_batch = MessageBatch(
                            data=raw_message,
                            batch_id=f"auto_wrapped_{id(raw_message)}",
                            metadata={'auto_wrapped': True},
                        )

                    if message_batch.is_empty:
                        return  # No message to process

                # Call subclass's specific message handling logic, passing message batch
                await self._handle_message(message_batch)
                return  # Successfully processed, return directly

            except Exception as e:
                last_error = e

                # Check if it's a fatal error and retry on fatal is disabled
                if (
                    self._error_handler.is_fatal_error(e)
                    and not retry_config.retry_on_fatal
                ):
                    self.logger.error(
                        "Fatal error in consumer %s, not retrying: %s",
                        self.job_id,
                        str(e),
                    )
                    raise

                # Check if it's a non-retryable error
                if (
                    not self._error_handler.is_retryable_error(e)
                    and not retry_config.retry_on_fatal
                ):
                    self.logger.error(
                        "Non-retryable error in consumer %s: %s", self.job_id, str(e)
                    )
                    raise

                if attempt < retry_config.max_retries:
                    # Calculate retry delay
                    delay = self._calculate_retry_delay(attempt, retry_config)

                    self.logger.warning(
                        "Message processing failed (attempt %d/%d) in consumer %s, retrying in %ss: %s",
                        attempt + 1,
                        retry_config.max_retries + 1,
                        self.job_id,
                        delay,
                        str(e),
                    )

                    await asyncio.sleep(delay)
                else:
                    # Maximum retries reached, raise exception
                    self.logger.error(
                        "Message processing failed after %d attempts in consumer %s: %s",
                        retry_config.max_retries + 1,
                        self.job_id,
                        str(e),
                    )
                    raise last_error

        # Theoretically unreachable, but for safety
        if last_error:
            raise last_error from None

    def _calculate_retry_delay(self, attempt: int, retry_config) -> float:
        """
        Calculate retry delay with support for exponential backoff and jitter

        Args:
            attempt: Current retry count (starting from 0)
            retry_config: Retry configuration

        Returns:
            float: Delay time in seconds
        """
        if retry_config.exponential_backoff:
            # Exponential backoff
            delay = retry_config.retry_delay * (
                retry_config.backoff_multiplier**attempt
            )
        else:
            # Fixed delay
            delay = retry_config.retry_delay

        # Limit maximum delay
        delay = min(delay, retry_config.max_delay)

        # Add random jitter
        if retry_config.jitter:
            # Random between 50% to 150%
            jitter_factor = 0.5 + random.random()
            delay = delay * jitter_factor

        return delay

    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        stats = self.stats.copy()
        stats['status'] = self.status.value
        stats['uptime'] = None

        if stats['start_time']:
            uptime = get_now_with_timezone() - stats['start_time']
            stats['uptime'] = uptime.total_seconds()

        return stats

    @abstractmethod
    async def _initialize(self) -> None:
        """
        Initialize resources
        Subclasses need to implement this method to initialize specific resources (e.g., connections, queues, etc.)
        """

    @abstractmethod
    async def _cleanup(self) -> None:
        """
        Cleanup resources
        Subclasses need to implement this method to clean up specific resources
        """

    @abstractmethod
    async def _has_messages(self) -> bool:
        """
        Check if there are messages available for consumption
        Subclasses need to implement this method to check if the message source has new messages

        Returns:
            bool: Whether there are messages available for consumption
        """

    @abstractmethod
    async def _fetch_message(self) -> Optional[Any]:
        """
        Fetch message data
        Subclasses need to implement this method to retrieve messages from the message source, can return any type of data
        The framework will automatically determine the type and wrap it if it's not a MessageBatch

        Returns:
            Optional[Any]: Retrieved message data, can be any type, return None if no message
        """

    @abstractmethod
    async def _handle_message(self, message_batch: MessageBatch) -> None:
        """
        Specific logic to handle message batch
        Subclasses need to implement this method to define the specific message processing logic

        Args:
            message_batch: Message batch to be processed, returned by _fetch_message

        Note:
            This method should process the incoming message batch, and throw an exception if processing fails
            Retry logic is handled by the base class, and the same message batch will be passed during retries
            Subclasses can access all messages via message_batch.messages and decide how to process them (individually or in batch)
        """
