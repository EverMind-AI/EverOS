"""StrategyRegistry — registration + DAG cycle detection.

Mutated at startup via :meth:`register` / :meth:`validate`, and at
runtime via :meth:`replace` (config hot-reload). Cycle detection is a
Kahn-style topological pass on the event-flow DAG implied by
``trigger.on`` (incoming) and ``emits`` (outgoing).
"""

from __future__ import annotations

from collections import defaultdict, deque

from everos.infra.ome.decorator import Strategy, StrategyMeta
from everos.infra.ome.events import BaseEvent, CronTick, IdleTick
from everos.infra.ome.exceptions import StartupValidationError
from everos.infra.ome.triggers import Cron, Idle, Immediate, Trigger


class StrategyRegistry:
    """Startup-time registry for offline strategies with cycle detection."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyMeta] = {}

    def register(self, strategy: Strategy) -> None:
        """Register a :class:`Strategy` returned by ``@offline_strategy``.

        Raises:
            StartupValidationError: If ``strategy`` is not a Strategy
                instance or its name is already registered.
        """
        if not isinstance(strategy, Strategy):
            label = getattr(strategy, "__name__", repr(strategy))
            raise StartupValidationError(
                f"register: {label} is not decorated with @offline_strategy"
            )
        meta = strategy.meta
        if meta.name in self._strategies:
            raise StartupValidationError(
                f"register: duplicate strategy name {meta.name!r}"
            )
        self._strategies[meta.name] = meta

    def replace(self, name: str, new_meta: StrategyMeta) -> None:
        """Swap an already-registered strategy's meta in place (hot-reload entry).

        Cycle / gate validation is **not** re-run; callers (currently
        :func:`apply_overrides`) must only feed metas where the
        DAG-shaping fields (``trigger.on``, ``emits``, trigger type)
        match the original. Raises ``KeyError`` if ``name`` is not yet
        registered.
        """
        if name not in self._strategies:
            raise KeyError(name)
        self._strategies[name] = new_meta

    def get(self, name: str) -> StrategyMeta:
        """Return meta by name (raises ``KeyError`` if absent)."""
        return self._strategies[name]

    def all(self) -> list[StrategyMeta]:
        """Return a snapshot list of every registered strategy."""
        return list(self._strategies.values())

    def lookup_by_event(self, event_cls: type[BaseEvent]) -> list[StrategyMeta]:
        """Return strategies that may receive an event of ``event_cls``.

        Resolution:
          * ``Immediate`` strategy listening on the class → match
          * ``CronTick``  → all Cron strategies (narrowed later by name)
          * ``IdleTick``  → all Idle strategies (narrowed later by name)

        Engine-emitted ticks carry a ``strategy_name`` field; dispatcher
        narrows the returned set to the single target via ``_routes_to``.
        """
        out: list[StrategyMeta] = []
        for m in self._strategies.values():
            if (
                (isinstance(m.trigger, Immediate) and event_cls in m.trigger.on)
                or (isinstance(m.trigger, Cron) and event_cls is CronTick)
                or (isinstance(m.trigger, Idle) and event_cls is IdleTick)
            ):
                out.append(m)
        return out

    def validate(self) -> None:
        """Validate the strategy DAG for cycles and gate field existence."""
        self._validate_no_cycles()
        self._validate_gate_event_fields()

    def _validate_no_cycles(self) -> None:
        """Kahn topological sort over the event-flow DAG.

        Edge ``s_a → s_b`` exists iff ``s_a.emits`` intersects
        ``s_b.trigger.on``.
        """
        adj: dict[str, set[str]] = defaultdict(set)
        indeg: dict[str, int] = dict.fromkeys(self._strategies, 0)

        for src in self._strategies.values():
            for ev in src.emits:
                for dst in self._strategies.values():
                    if (
                        isinstance(dst.trigger, Immediate)
                        and ev in dst.trigger.on
                        and dst.name not in adj[src.name]
                    ):
                        adj[src.name].add(dst.name)
                        indeg[dst.name] += 1

        queue = deque(n for n, d in indeg.items() if d == 0)
        visited = 0
        while queue:
            n = queue.popleft()
            visited += 1
            for nbr in adj[n]:
                indeg[nbr] -= 1
                if indeg[nbr] == 0:
                    queue.append(nbr)

        if visited < len(self._strategies):
            raise StartupValidationError("cycle detected in strategy DAG")

    def _validate_gate_event_fields(self) -> None:
        """Reject any ``gate.event_field`` missing from a receivable event class.

        Without this check a typo silently collapses every event into one
        shared bucket and the rate gate stops segmenting.
        """
        for meta in self._strategies.values():
            if meta.gate is None or meta.gate.event_field is None:
                continue
            field = meta.gate.event_field
            for ev_cls in _event_classes_for_trigger(meta.trigger):
                if field not in ev_cls.model_fields:  # type: ignore[operator]  # Pydantic model_fields → dict via @deprecated_instance_property (pydantic/main.py:277)
                    raise StartupValidationError(
                        f"strategy {meta.name!r}: gate.event_field {field!r} "
                        f"not found in {ev_cls.__name__} fields "
                        f"(available: {list(ev_cls.model_fields)})"  # type: ignore[arg-type]  # same as above
                    )


def _event_classes_for_trigger(trigger: Trigger) -> list[type[BaseEvent]]:
    """Enumerate event classes a strategy with the given trigger receives."""
    if isinstance(trigger, Immediate):
        return list(trigger.on)
    if isinstance(trigger, Cron):
        return [CronTick]
    if isinstance(trigger, Idle):
        return [IdleTick]
    raise NotImplementedError(f"unknown trigger type: {type(trigger).__name__}")
