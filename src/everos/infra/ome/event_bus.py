"""The :class:`~everos.memory.ports.EventBus` backed by an OME engine."""

from __future__ import annotations

__all__ = ["OmeEventBus"]

from collections.abc import Sequence

from everos.infra.ome.engine import OfflineEngine
from everos.memory.ports import Event


class OmeEventBus:
    """Publishes events by dispatching them through ``engine``.

    Args:
        engine: The started engine whose strategies receive the events.
    """

    def __init__(self, engine: OfflineEngine) -> None:
        self._engine = engine

    async def publish(self, events: Sequence[Event]) -> None:
        """Dispatch ``events`` one by one, in order."""
        for event in events:
            await self._engine.emit(event)
