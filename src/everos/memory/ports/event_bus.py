"""The port through which the application publishes domain events.

Domain code never publishes: pipelines and strategies return the events
they produce, and the caller that orchestrates them — the service layer, or
the scheduler adapter for strategy runs — hands them to an ``EventBus``.
"""

from __future__ import annotations

__all__ = ["EventBus"]

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from everos.memory.ports.events import Event


@runtime_checkable
class EventBus(Protocol):
    """Delivers events to the strategies subscribed to them."""

    async def publish(self, events: Sequence[Event]) -> None:
        """Publish ``events`` in order."""
        ...
