"""IdleScanner — periodic scan of idle_store, emits IdleTick for overdue buckets."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from everos.component.utils.datetime import get_utc_now
from everos.core.observability.logging import get_logger
from everos.infra.ome._stores.idle import IdleStore
from everos.infra.ome.events import BaseEvent, IdleTick
from everos.infra.ome.triggers import Idle

logger = get_logger(__name__)


class IdleScanner:
    """Scans idle_store for overdue buckets and emits IdleTick events."""

    def __init__(
        self,
        *,
        strategy_name: str,
        trigger: Idle,
        idle_store: IdleStore,
        emit: Callable[[BaseEvent], Awaitable[None]],
    ) -> None:
        self._name = strategy_name
        self._trigger = trigger
        self._idle_store = idle_store
        self._emit = emit

    async def scan_once(self, *, now: datetime | None = None) -> None:
        """Find overdue buckets and emit IdleTick for each.

        Per-bucket emit failures are caught and logged so a single
        downstream error (e.g. dispatch hitting a transient DB lock)
        cannot prevent sibling buckets from being notified this round.
        """
        effective_now = now if now is not None else get_utc_now()
        overdue = await self._idle_store.scan_idle(
            self._name,
            idle_seconds=self._trigger.idle_seconds,
            now=effective_now,
        )
        for bucket_key in overdue:
            try:
                await self._emit(
                    IdleTick(
                        strategy_name=self._name,
                        bucket_key=bucket_key,
                        idle_seconds=self._trigger.idle_seconds,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "idle_emit_failed",
                    strategy_name=self._name,
                    bucket_key=bucket_key,
                )
