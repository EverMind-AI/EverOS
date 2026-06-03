"""In-memory test doubles for the OME StrategyContext Protocol.

Use FakeStrategyContext when you want to unit-test a strategy function
in isolation without spinning up a full OfflineEngine.
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger
from everos.infra.ome.events import BaseEvent


class FakeStrategyContext:
    """Implements StrategyContext Protocol; collects emit() calls in a list.

    Attributes:
        run_id: Unique identifier for this run (default: "fake_run").
        logger: A structlog BoundLogger for test logging.
        emitted: List of BaseEvent objects passed to emit().
    """

    def __init__(self, *, run_id: str = "fake_run") -> None:
        """Initialize a FakeStrategyContext.

        Args:
            run_id: Run identifier, defaults to "fake_run".
        """
        self.run_id = run_id
        self.logger = get_logger("ome.fake_ctx")
        self.emitted: list[BaseEvent] = []

    async def emit(self, event: BaseEvent) -> None:
        """Collect an event into the emitted list.

        Args:
            event: The BaseEvent to emit.
        """
        self.emitted.append(event)
