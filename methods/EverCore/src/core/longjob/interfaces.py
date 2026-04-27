"""
Long job interfaces and base classes.
Long task interfaces and base class definitions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import asyncio
from enum import Enum
from dataclasses import dataclass

# Import error classes from longjob_error
from core.longjob.longjob_error import FatalError, BusinessLogicError


@dataclass
class MessageBatch:
    """
    Message wrapper class
    Uniformly encapsulates message data, without restricting specific types (can be a single message, list, or any business-defined structure)
    """

    data: Any  # Message data, can be any type: single message, list, dictionary, etc.
    batch_id: Optional[str] = None  # Batch ID, used for tracking and logging
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata information

    def __post_init__(self):
        """Post-initialization processing"""
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_empty(self) -> bool:
        """Check if data is empty"""
        if self.data is None:
            return True

        # If it's a list or similar container, check length
        if hasattr(self.data, '__len__'):
            try:
                return len(self.data) == 0
            except (TypeError, AttributeError):
                pass

        return False


class LongJobStatus(Enum):
    """Long job status enumeration"""

    IDLE = "idle"  # Idle state
    STARTING = "starting"  # Starting
    RUNNING = "running"  # Running
    STOPPING = "stopping"  # Stopping
    STOPPED = "stopped"  # Stopped
    ERROR = "error"  # Error state


class LongJobInterface(ABC):
    """
    Long job interface definition.
    All long jobs need to implement this interface.
    """

    def __init__(self, job_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize long job

        Args:
            job_id: Job ID, used for identification and management
            config: Job configuration parameters
        """
        self.job_id = job_id
        self.config = config or {}
        self.status = LongJobStatus.IDLE
        self._stop_event = asyncio.Event()

    @abstractmethod
    async def start(self) -> None:
        """
        Start long job
        Implementation classes need to start the specific work logic here
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shut down long job
        Implementation classes need to clean up resources and stop work here
        """

    def get_status(self) -> LongJobStatus:
        """Get current job status"""
        return self.status

    def is_running(self) -> bool:
        """Check if the job is currently running"""
        return self.status == LongJobStatus.RUNNING

    def should_stop(self) -> bool:
        """Check if the job should stop"""
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Request to stop the job"""
        self._stop_event.set()


class ErrorHandler(ABC):
    """
    Error handler interface
    Used to handle exceptions during long job execution
    """

    @abstractmethod
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Handle error

        Args:
            error: The exception that occurred
            context: Error context information

        Returns:
            bool: True means execution can continue, False means it should stop
        """

    def is_fatal_error(self, error: Exception) -> bool:
        """
        Determine if it is a fatal error

        Args:
            error: Exception instance

        Returns:
            bool: True means fatal error, should not retry
        """
        # Check if it's an explicitly marked fatal error
        if isinstance(error, FatalError):
            return True

        # Check common fatal error types
        fatal_error_types = (
            MemoryError,
            SystemExit,
            KeyboardInterrupt,
            ImportError,
            SyntaxError,
            TypeError,  # Usually indicates programming errors
            AttributeError,  # Usually indicates programming errors
        )

        return isinstance(error, fatal_error_types)

    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if it is a retryable error

        Args:
            error: Exception instance

        Returns:
            bool: True means it can be retried
        """
        # If it's a fatal error, not retryable
        if self.is_fatal_error(error):
            return False

        # Explicitly marked business logic errors can be retried
        if isinstance(error, BusinessLogicError):
            return True

        # Network-related errors are usually retryable
        retryable_error_types = (
            ConnectionError,
            TimeoutError,
            OSError,
        )  # Includes network errors

        return isinstance(error, retryable_error_types)


class RetryConfig:
    """Retry configuration"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
        max_delay: float = 60.0,
        jitter: bool = True,
        backoff_multiplier: float = 2.0,
        retry_on_fatal: bool = False,
    ):
        """
        Initialize retry configuration

        Args:
            max_retries: Maximum number of retries
            retry_delay: Initial retry delay time (seconds)
            exponential_backoff: Whether to use exponential backoff
            max_delay: Maximum delay time (seconds)
            jitter: Whether to add random jitter
            backoff_multiplier: Exponential backoff multiplier
            retry_on_fatal: Whether to retry on fatal errors (usually False)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.max_delay = max_delay
        self.jitter = jitter
        self.backoff_multiplier = backoff_multiplier
        self.retry_on_fatal = retry_on_fatal


class ConsumerConfig:
    """Consumer configuration"""

    def __init__(
        self,
        timeout: float = 600.0,
        retry_config: Optional[RetryConfig] = None,
        error_handler: Optional[ErrorHandler] = None,
    ):
        """
        Initialize consumer configuration

        Args:
            timeout: Timeout for consuming a single message (seconds), including retries
            retry_config: Retry configuration
            error_handler: Error handler
        """
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.error_handler = error_handler
