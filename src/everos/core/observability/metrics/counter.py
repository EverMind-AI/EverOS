"""Counter wrapper around ``prometheus_client.Counter``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prometheus_client import Counter as PromCounter

from .registry import get_metrics_registry


class Counter:
    """Monotonically-increasing counter (totals, error counts)."""

    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = "",
        subsystem: str = "",
        unit: str = "",
    ) -> None:
        self._counter = PromCounter(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=get_metrics_registry(),
        )
        self._labelnames = tuple(labelnames)

    def labels(self, **labels: str) -> LabeledCounter:
        return LabeledCounter(self._counter.labels(**labels))

    def inc(self, amount: float = 1.0) -> None:
        self._counter.inc(amount)


class LabeledCounter:
    """Counter slice with labels applied."""

    def __init__(self, labeled: Any) -> None:
        self._labeled = labeled

    def inc(self, amount: float = 1.0) -> None:
        self._labeled.inc(amount)
