"""Runner — single-strategy execution with attempt-level retry + DLQ.

Acquires ``engine_sem`` (FIFO), drives the per-attempt RunRecord state
machine (``RUNNING → SUCCESS / FAILED / DEAD_LETTER``), and fires
``on_dead_letter`` after exhausted retries or contract violations.

Per attempt, binds ``strategy_name`` / ``run_id`` / ``attempt`` into
``structlog.contextvars`` (so every log record carries those fields
automatically) and sets ``_CURRENT_STRATEGY`` ContextVar around
``meta.func`` (so ``engine.emit`` can refuse direct calls from inside
a strategy — strategies emit via ``ctx.emit``).

**Idempotency contract**: if ``mark_success`` / ``mark_failed`` /
``mark_dead_letter`` fails after the strategy body returned, the
``RUNNING`` row stays and crash recovery on next start will treat the
run as crashed and re-enqueue the same event. Strategy bodies must
therefore be safe to re-execute with the same payload.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from structlog.contextvars import bound_contextvars

from everos.component.utils.datetime import get_utc_now
from everos.core.observability.logging import get_logger
from everos.infra.ome._dispatch._state import _CURRENT_STRATEGY
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome.decorator import StrategyMeta
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.exceptions import EmitNotDeclaredError, StrategyContractError
from everos.infra.ome.records import RunRecord

if TYPE_CHECKING:
    from everos.infra.ome.engine import OfflineEngine

logger = get_logger(__name__)


class _RunCtx:
    """Implements :class:`~everos.infra.ome.context.StrategyContext` Protocol.

    Carries ``run_id``, a strategy-scoped logger, the ``emit``
    callback that enforces the declared ``emits=[...]`` contract,
    and engine-delegated helpers for event/run queries.
    """

    def __init__(
        self,
        *,
        run_id: str,
        strategy_name: str,
        emit_hook: Callable[[BaseEvent], Awaitable[None]],
        declared_emits: frozenset[type[BaseEvent]],
        engine: OfflineEngine,
    ) -> None:
        self.run_id = run_id
        self.logger = get_logger("ome.strategy")
        self._emit_hook = emit_hook
        self._declared = declared_emits
        self._strategy_name = strategy_name
        self._engine = engine

    async def emit(self, event: BaseEvent) -> None:
        if type(event) not in self._declared:
            raise EmitNotDeclaredError(
                strategy=self._strategy_name,
                event=event,
            )
        await self._emit_hook(event)

    async def wait_for_event(
        self,
        event_id: str,
        *,
        timeout: float = 120.0,  # noqa: ASYNC109
    ) -> list[RunRecord]:
        """Poll until all runs for ``event_id`` reach a terminal status."""
        return await self._engine.wait_for_event(event_id, timeout=timeout)

    async def list_runs_by_event_id(self, event_id: str) -> list[RunRecord]:
        """Return all run records triggered by ``event_id``."""
        return await self._engine.list_runs_by_event_id(event_id)


class Runner:
    """Drive one strategy invocation through retries to a terminal state."""

    def __init__(
        self,
        *,
        run_record_store: RunRecordStore,
        engine_sem: asyncio.Semaphore,
        emit_hook: Callable[[BaseEvent], Awaitable[None]],
        on_dead_letter: Callable[[RunRecord], None] | None = None,
        engine: OfflineEngine,
    ) -> None:
        self._rec = run_record_store
        self._sem = engine_sem
        self._emit_hook = emit_hook
        self._on_dead_letter = on_dead_letter
        self._engine = engine

    async def run(
        self,
        meta: StrategyMeta,
        event: BaseEvent,
        *,
        run_id: str,
        max_retries_snapshot: int,
    ) -> None:
        """Execute ``meta.func(event, ctx)`` with the attempt retry loop.

        Holds ``engine_sem`` for the full retry chain so concurrency cap
        applies end-to-end. Each attempt gets a fresh ``run_id`` after
        the first, so the run history records every try.
        """
        if max_retries_snapshot < 0:
            raise ValueError(
                f"max_retries_snapshot must be >= 0, got {max_retries_snapshot}"
            )

        async with self._sem:
            event_topic = type(event).topic()
            event_payload = event.model_dump_json()
            current_run_id = run_id

            for attempt in range(max_retries_snapshot + 1):
                if attempt > 0:
                    current_run_id = uuid4().hex
                terminated = await self._run_one_attempt(
                    meta=meta,
                    event=event,
                    current_run_id=current_run_id,
                    attempt=attempt,
                    event_topic=event_topic,
                    event_payload=event_payload,
                    max_retries_snapshot=max_retries_snapshot,
                )
                if terminated:
                    return

    async def _run_one_attempt(
        self,
        *,
        meta: StrategyMeta,
        event: BaseEvent,
        current_run_id: str,
        attempt: int,
        event_topic: str,
        event_payload: str,
        max_retries_snapshot: int,
    ) -> bool:
        """Run one attempt; return ``True`` if a terminal state was
        written (success / dead-letter or persistence failure), ``False``
        if FAILED and the caller should loop into the next attempt.
        """
        ctx = _RunCtx(
            run_id=current_run_id,
            strategy_name=meta.name,
            emit_hook=self._emit_hook,
            declared_emits=meta.emits,
            engine=self._engine,
        )
        with bound_contextvars(  # type: ignore[arg-type]  # structlog typed as Generator; @contextmanager wraps at runtime (structlog/contextvars.py:170)
            strategy_name=meta.name,
            run_id=current_run_id,
            attempt=attempt,
        ):
            if not await self._record_start(
                run_id=current_run_id,
                strategy_name=meta.name,
                attempt=attempt,
                event_topic=event_topic,
                event_payload=event_payload,
                max_retries_snapshot=max_retries_snapshot,
                event_id=event.event_id,
            ):
                return True  # mark_running failed; abort run, no DB row exists
            try:
                token = _CURRENT_STRATEGY.set(meta)
                try:
                    await meta.func(event, ctx)
                finally:
                    _CURRENT_STRATEGY.reset(token)
            except StrategyContractError as e:
                await self._terminate_dead_letter(current_run_id, _format_error(e))
                return True
            except Exception as e:  # noqa: BLE001
                err = _format_error(e)
                if attempt < max_retries_snapshot:
                    await self._rec.mark_failed(
                        run_id=current_run_id,
                        finished_at=get_utc_now(),
                        error=err,
                    )
                    return False  # caller will retry
                await self._terminate_dead_letter(current_run_id, err)
                return True
            else:
                await self._rec.mark_success(
                    run_id=current_run_id,
                    finished_at=get_utc_now(),
                )
                return True

    async def _record_start(
        self,
        *,
        run_id: str,
        strategy_name: str,
        attempt: int,
        event_topic: str,
        event_payload: str,
        max_retries_snapshot: int,
        event_id: str,
    ) -> bool:
        """Persist this attempt as RUNNING; return ``False`` on write failure.

        When the write fails (DB lock, disk full, ...) the caller
        aborts the retry loop — without a RUNNING row crash recovery
        cannot rediscover the run, and it is silently lost. The
        exception log emitted here is the only audit trail.
        """
        try:
            await self._rec.mark_running(
                run_id=run_id,
                strategy_name=strategy_name,
                attempt=attempt,
                event_topic=event_topic,
                event_payload=event_payload,
                max_retries_snapshot=max_retries_snapshot,
                event_id=event_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "mark_running_failed",
                run_id=run_id,
                strategy_name=strategy_name,
                attempt=attempt,
            )
            return False
        return True

    async def _terminate_dead_letter(self, run_id: str, error: str) -> None:
        """Mark DEAD_LETTER and fire ``on_dead_letter`` callback if set."""
        await self._rec.mark_dead_letter(
            run_id=run_id,
            finished_at=get_utc_now(),
            error=error,
        )
        await self._fire_dead_letter_callback(run_id)

    async def _fire_dead_letter_callback(self, run_id: str) -> None:
        if self._on_dead_letter is None:
            return
        rec = await self._rec.get(run_id)
        if rec is None:
            return
        try:
            self._on_dead_letter(rec)
        except Exception:  # noqa: BLE001
            logger.exception("on_dead_letter_failed")


def _format_error(e: BaseException) -> str:
    """Format an exception with type, message, and full traceback."""
    return f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
