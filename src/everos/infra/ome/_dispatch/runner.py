"""Runner — single-strategy execution with attempt-level retry + DLQ.

Acquires ``engine_sem`` (FIFO), drives the per-attempt RunRecord state
machine (``RUNNING → SUCCESS / FAILED / DEAD_LETTER``), and fires
``on_dead_letter`` after exhausted retries or contract violations.

Per attempt, binds ``strategy_name`` / ``run_id`` / ``attempt`` into
the log context (so every log record carries those fields
automatically) and sets ``_CURRENT_STRATEGY`` ContextVar around
``meta.func`` (so ``engine.emit`` can refuse direct calls from inside
a strategy — strategies return their follow-up events instead).

**Idempotency contract**: if ``mark_success`` / ``mark_failed`` /
``mark_dead_letter`` fails after the strategy body returned, the
``RUNNING`` row stays and crash recovery on next start will treat the
run as crashed and re-enqueue the same event. Strategy bodies must
therefore be safe to re-execute with the same payload.
"""

from __future__ import annotations

import asyncio
import random
import traceback
from collections.abc import Awaitable, Callable
from uuid import uuid4

from everos.core.observability.logging import bound_log_context, get_logger
from everos.core.observability.tracing import memory_span, use_traceparent
from everos.core.utils.datetime import get_utc_now
from everos.infra.ome._dispatch._state import _CURRENT_STRATEGY
from everos.infra.ome._dispatch.strategy import RunContext, StrategyMeta
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.exceptions import StrategyContractError
from everos.infra.ome.records import RunRecord
from everos.memory.ports import Event

logger = get_logger(__name__)


class Runner:
    """Drive one strategy invocation through retries to a terminal state."""

    def __init__(
        self,
        *,
        run_record_store: RunRecordStore,
        engine_sem: asyncio.Semaphore,
        emit_hook: Callable[[Event], Awaitable[None]],
        config: OMEConfig,
        on_dead_letter: Callable[[RunRecord], None] | None = None,
    ) -> None:
        self._rec = run_record_store
        self._sem = engine_sem
        self._emit_hook = emit_hook
        self._config = config
        self._on_dead_letter = on_dead_letter

    async def run(
        self,
        meta: StrategyMeta,
        event: Event,
        *,
        run_id: str,
        max_retries_snapshot: int,
        traceparent: str | None = None,
    ) -> None:
        """Execute ``meta.func(event, ctx)`` with the attempt retry loop.

        ``engine_sem`` is held per *attempt*, not across the whole retry
        chain: the backoff sleep happens outside it. The cap exists to
        bound concurrent strategy work — LLM calls, embeddings, storage
        IO — and a coroutine sleeping between attempts consumes none of
        that. Holding the slot across the sleep turned a partial outage
        into a total stall: with ``max_concurrent_runs`` slots and a
        ``1s → 2s → 4s`` backoff, enough simultaneously-failing runs
        park every slot in ``asyncio.sleep`` and starve strategies that
        would have succeeded. Backpressure on the failing work is
        wanted; backpressure on everything else is not.

        Each attempt gets a fresh ``run_id`` after the first, so the run
        history records every try.
        """
        if max_retries_snapshot < 0:
            raise ValueError(
                f"max_retries_snapshot must be >= 0, got {max_retries_snapshot}"
            )

        event_topic = type(event).topic()
        event_payload = event.model_dump_json()
        current_run_id = run_id

        for attempt in range(max_retries_snapshot + 1):
            if attempt > 0:
                await self._sleep_backoff(attempt)
                current_run_id = uuid4().hex
            async with self._sem:
                terminated = await self._run_one_attempt(
                    meta=meta,
                    event=event,
                    current_run_id=current_run_id,
                    attempt=attempt,
                    event_topic=event_topic,
                    event_payload=event_payload,
                    max_retries_snapshot=max_retries_snapshot,
                    traceparent=traceparent,
                )
            if terminated:
                return

    async def _sleep_backoff(self, attempt: int) -> None:
        """Sleep before retry ``attempt`` (1-indexed): ``base * 2**(attempt-1)``,
        capped at ``retry_backoff_cap_seconds``, plus up to
        ``retry_jitter_seconds`` of uniform jitter. ``retry_backoff_base_seconds
        == 0.0`` disables backoff entirely (used by tests that don't
        monkeypatch ``asyncio.sleep``).
        """
        base = self._config.retry_backoff_base_seconds
        if base <= 0.0:
            return
        cap = self._config.retry_backoff_cap_seconds
        jitter_max = self._config.retry_jitter_seconds
        sleep_seconds = min(base * (2 ** (attempt - 1)), cap)
        if jitter_max > 0.0:
            sleep_seconds += random.uniform(0.0, jitter_max)
        await asyncio.sleep(sleep_seconds)

    async def _run_one_attempt(
        self,
        *,
        meta: StrategyMeta,
        event: Event,
        current_run_id: str,
        attempt: int,
        event_topic: str,
        event_payload: str,
        max_retries_snapshot: int,
        traceparent: str | None = None,
    ) -> bool:
        """Run one attempt; return ``True`` if a terminal state was
        written (success / dead-letter or persistence failure), ``False``
        if FAILED and the caller should loop into the next attempt.
        """
        ctx = RunContext(
            strategy_name=meta.name,
            declared_emits=meta.emits,
            emit_hook=self._emit_hook,
        )
        with bound_log_context(
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
                    # Continue the triggering request's trace when a traceparent
                    # was carried across the APScheduler boundary; otherwise the
                    # agent span roots its own trace (e.g. cron / recovery).
                    with (
                        use_traceparent(traceparent),
                        memory_span(
                            f"everos.ome.{meta.name}",
                            observation_type="agent",
                            metadata={
                                "strategy": meta.name,
                                "run_id": current_run_id,
                                "attempt": attempt,
                                "event_topic": event_topic,
                            },
                        ),
                    ):
                        # Timeout INSIDE the span so a killed attempt is still
                        # attributed to the strategy that hung. TimeoutError is an
                        # Exception, so it lands in the retry / dead-letter handler
                        # below like any other failure -- the slot is released, the
                        # record leaves RUNNING, and the work is retried rather than
                        # silently abandoned.
                        timeout = self._config.run_timeout_seconds
                        if timeout is None:
                            await meta.func(event, ctx)
                        else:
                            async with asyncio.timeout(timeout):
                                await meta.func(event, ctx)
                finally:
                    _CURRENT_STRATEGY.reset(token)
            except StrategyContractError as e:
                await self._terminate_dead_letter(current_run_id, _format_error(e))
                return True
            except Exception as e:
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
        except Exception:
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
        except Exception:
            logger.exception("on_dead_letter_failed")


def _format_error(e: BaseException) -> str:
    """Format an exception with type, message, and full traceback."""
    return f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
