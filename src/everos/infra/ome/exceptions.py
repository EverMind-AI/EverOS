"""OME exception hierarchy."""

from __future__ import annotations

from everos.memory.ports import Event


class OMEError(Exception):
    """Base for all OME-internal errors."""


class StartupValidationError(OMEError):
    """Raised by engine.start() for any startup-time validation failure."""


class EngineLockHeldError(OMEError):
    """Raised when another OfflineEngine instance holds the jobstore lock."""


class StrategyContractError(OMEError):
    """Base for strategy-side contract violations.

    Subclasses indicate a programming bug in the strategy code that no
    retry can fix (wrong API usage, undeclared emit). Runner
    short-circuits the attempt loop on these and dead-letters
    immediately — consuming the retry budget would only delay the
    inevitable and spam logs. External callers can ``except
    StrategyContractError`` to handle the whole category at once.
    """


class EngineCallFromStrategyError(StrategyContractError):
    """A strategy called a public OfflineEngine method directly.

    The convention is: a strategy never touches the engine — it returns
    its follow-up events and OME publishes them. Engine methods
    (``emit``, ``trigger_manual``, ``inspect_dispatch``, ``list_runs``,
    ``get_run_status``, ``reschedule_*``) are for external callers —
    strategies invoking them bypass the framework's contracts.
    """

    def __init__(self, strategy: str, method: str) -> None:
        self.strategy = strategy
        self.method = method
        super().__init__(
            f"strategy {strategy!r} called engine.{method}() directly; "
            "strategies return their follow-up events instead of calling "
            "the engine"
        )


class EmitNotDeclaredError(StrategyContractError):
    """Raised when a strategy returns an event its spec does not declare."""

    def __init__(self, strategy: str, event: Event) -> None:
        self.strategy = strategy
        self.event = event
        super().__init__(
            f"strategy {strategy!r} emitted {type(event).__name__!r} "
            "which is not in its declared emits"
        )
