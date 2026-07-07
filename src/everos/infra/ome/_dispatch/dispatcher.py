"""EventDispatcher — routing layer applying the three OME gates.

For each dispatched event, every candidate strategy is run through three
gates in order:

  1. ``enabled``     — strategy may be hot-disabled via config
  2. ``applies_to``  — per-strategy predicate over the event payload
  3. ``Counter``     — N-of-M rate/threshold gate against
                       :class:`CounterStore`

:meth:`dispatch` is the read-write entry point — passing the counter
gate increments the counter and returns ``(meta, run_id)`` pairs to
enqueue. :meth:`inspect` is its dry-run twin — same gates, no counter
mutation; returns one :class:`StrategyRouteInfo` per matched strategy
including a snapshot of the counter so debug callers can see why a
strategy will or won't fire.

By design ``inspect`` does not accept ``force_enabled`` /
``strategy_filter``: those are runtime overrides for the routing side
(``trigger_manual``), not properties a debugger should second-guess.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from everos.core.observability.logging import get_logger
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome._stores.counter import CounterStore
from everos.infra.ome.decorator import StrategyMeta
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.records import CounterProgress, StrategyRouteInfo

logger = get_logger(__name__)


class EventDispatcher:
    """Apply ``enabled / applies_to / Counter`` gates to one event."""

    def __init__(
        self,
        *,
        registry: StrategyRegistry,
        counter_store: CounterStore,
    ) -> None:
        self._registry = registry
        self._counter_store = counter_store

    async def dispatch(
        self,
        event: BaseEvent,
        *,
        force_enabled: bool = False,
        strategy_filter: str | None = None,
    ) -> list[tuple[StrategyMeta, str]]:
        """Run gates and return ``(meta, run_id)`` pairs to enqueue.

        Args:
            event: The event to route.
            force_enabled: Bypass the ``meta.enabled`` gate. ``applies_to``
                and the counter still apply. Used by manual triggers
                with ``force=True``.
            strategy_filter: Restrict to one strategy name regardless of
                whether it subscribes to ``type(event)``. Manual triggers
                use this when targeting a strategy with a caller-supplied
                event. Raises ``KeyError`` if the name is not registered.

        ``applies_to`` callables raised by a single strategy are caught,
        logged, and treated as ``False`` for that strategy alone — sibling
        strategies still dispatch. Framework errors (e.g. CounterStore
        I/O) propagate.
        """
        if strategy_filter is not None:
            metas: list[StrategyMeta] = [self._registry.get(strategy_filter)]
        else:
            metas = list(self._registry.lookup_by_event(type(event)))
        out: list[tuple[StrategyMeta, str]] = []
        for meta in metas:
            if not _routes_to(event, meta):
                continue
            if not force_enabled and not meta.enabled:
                continue
            if not _safe_applies(meta, event):
                continue
            if meta.gate is not None:
                bucket = _bucket_key(event, meta.gate.event_field)
                passed, _ = await self._counter_store.incr_and_check(
                    meta.name,
                    bucket,
                    threshold=meta.gate.threshold,
                    cooldown_seconds=meta.gate.cooldown_seconds,
                )
                if not passed:
                    continue
            out.append((meta, uuid4().hex))
        return out

    async def inspect(self, event: BaseEvent) -> list[StrategyRouteInfo]:
        """Dry-run twin of :meth:`dispatch` — no counter mutation.

        Returns one :class:`StrategyRouteInfo` per matched strategy with
        per-gate pass flags and a counter snapshot (read-only via
        ``get_progress``). Same exception policy as :meth:`dispatch`:
        a strategy's faulty ``applies_to`` callable is logged and that
        strategy reports ``applies_to_pass=False`` rather than tanking
        the whole inspection.
        """
        out: list[StrategyRouteInfo] = []
        for meta in self._registry.lookup_by_event(type(event)):
            if not _routes_to(event, meta):
                continue
            enabled_pass = bool(meta.enabled)
            applies_pass = enabled_pass and _safe_applies(meta, event)
            counter_pass = applies_pass and (meta.gate is None)
            progress: CounterProgress | None = None
            if applies_pass and meta.gate is not None:
                bucket = _bucket_key(event, meta.gate.event_field)
                cur = await self._counter_store.get_progress(
                    meta.name,
                    bucket,
                )
                next_cur = cur + 1
                progress = CounterProgress(
                    current=next_cur, threshold=meta.gate.threshold
                )
                counter_pass = next_cur >= meta.gate.threshold
            out.append(
                StrategyRouteInfo(
                    strategy_name=meta.name,
                    enabled_pass=enabled_pass,
                    applies_to_pass=applies_pass,
                    counter_pass=counter_pass,
                    counter_progress=progress,
                )
            )
        return out


def _routes_to(event: BaseEvent, meta: StrategyMeta) -> bool:
    """Narrow engine-emitted ticks to their single target strategy.

    Cron / Idle / Manual ticks carry a ``strategy_name`` naming the
    intended recipient — without this filter two strategies listening
    on the same tick class would cross-fire. Business events have no
    such field and therefore fan out to every matching strategy.
    """
    target = getattr(event, "strategy_name", None)
    return target is None or target == meta.name


def _safe_applies(meta: StrategyMeta, event: BaseEvent) -> bool:
    """Evaluate ``meta.applies_to`` with user-callable exceptions isolated.

    A faulty ``applies_to`` callable is logged at exception level with
    ``strategy_name`` + ``event_topic`` context and treated as
    ``False`` so that a single buggy predicate cannot tank the entire
    fan-out for an event.
    """
    try:
        return _applies(meta.applies_to, event)
    except Exception:
        logger.exception(
            "applies_to_callable_raised",
            strategy_name=meta.name,
            event_topic=type(event).topic(),
        )
        return False


def _applies(
    spec: str | Callable[[BaseEvent], bool] | None,
    event: BaseEvent,
) -> bool:
    """Resolve ``applies_to`` semantics.

      * ``None`` — strategy applies to every event in its subscription
      * callable — invoke and bool-cast the result
      * str — read the named event attribute and bool-cast it; falsy
        values (``""``, ``0``, ``None``, empty containers) are treated
        as "field unset", so the strategy does NOT apply

    Exceptions raised by a user callable propagate; the dispatcher wraps
    this call in :func:`_safe_applies` to localise blast radius.
    """
    if spec is None:
        return True
    if callable(spec):
        return bool(spec(event))
    return bool(getattr(event, spec, None))


def _bucket_key(event: BaseEvent, field: str | None) -> str:
    """Compute a Counter-store bucket key from an event field.

    ``field=None`` means the gate is un-bucketed → single shared bucket
    ``"__all__"``. Missing or ``None`` field values map to ``"__none__"``
    so a typo doesn't accidentally collapse every event into ``"__all__"``
    (the StrategyRegistry validator catches typos at startup; the sentinel
    here is the runtime safety net).
    """
    if field is None:
        return "__all__"
    val = getattr(event, field, None)
    return str(val) if val is not None else "__none__"
