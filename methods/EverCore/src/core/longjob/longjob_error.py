"""
Long job system specific errors.
Long task system specific error definitions.
"""


class FatalError(Exception):
    """
    Fatal error, should not retry
    Used to identify errors that cannot be resolved by retrying

    Examples:
        - Out of memory
        - System-level errors
        - Configuration errors
        - Programming errors (TypeError, AttributeError, etc.)
    """


class BusinessLogicError(Exception):
    """
    Business logic error, can retry
    Used to identify errors that might be resolved by retrying

    Examples:
        - Network connection errors
        - Temporary database connection issues
        - Third-party service temporarily unavailable
        - Resource lock conflicts
    """


class LongJobError(Exception):
    """
    Long task system base error
    Base class for all long task related errors
    """


class JobNotFoundError(LongJobError):
    """Job not found error"""


class JobAlreadyExistsError(LongJobError):
    """Job already exists error"""


class JobStateError(LongJobError):
    """Job state error"""


class ManagerShutdownError(LongJobError):
    """Manager has been shut down error"""


class MaxConcurrentJobsError(LongJobError):
    """Exceeded maximum number of concurrent jobs error"""
