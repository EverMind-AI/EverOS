"""In-memory test doubles for the OME StrategyContext Protocol.

Use FakeStrategyContext when you want to unit-test a strategy function
in isolation without spinning up a full OfflineEngine.
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.records import RunRecord


class FakeStrategyContext:
    """Implements StrategyContext Protocol; collects emit() calls in a list.

    Args:
        run_id: Run identifier, defaults to ``"fake_run"``.

    Attributes:
        run_id: Unique identifier for this run (default: ``"fake_run"``).
        logger: A structlog BoundLogger for test logging.
        emitted: List of BaseEvent objects passed to emit().
    """

    def __init__(self, *, run_id: str = "fake_run") -> None:
        self.run_id = run_id
        self.logger = get_logger("ome.fake_ctx")
        self.emitted: list[BaseEvent] = []

    async def emit(self, event: BaseEvent) -> None:
        """Collect an event into the emitted list.

        Args:
            event: The BaseEvent to emit.
        """
        self.emitted.append(event)

    async def wait_for_event(
        self,
        event_id: str,
        *,
        timeout: float = 120.0,  # noqa: ASYNC109
    ) -> list[RunRecord]:
        """No-op stub; returns an empty list.

        Args:
            event_id: The event identifier to wait for.
            timeout: Maximum seconds to wait (unused in fake).

        Returns:
            Empty list (no runs in test doubles).
        """
        return []

    async def list_runs_by_event_id(self, event_id: str) -> list[RunRecord]:
        """No-op stub; returns an empty list.

        Args:
            event_id: The event identifier to query.

        Returns:
            Empty list (no runs in test doubles).
        """
        return []
