"""Gauge wrapper around ``prometheus_client.Gauge``.

Async auto-refresh is intentionally not included in v0.1; subclass
:class:`Gauge` and call :meth:`set` from your own scheduling logic when
needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prometheus_client import Gauge as PromGauge

from .registry import get_metrics_registry


class Gauge:
    """Instantaneous numeric value (queue depth, cache size)."""

    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = "",
        subsystem: str = "",
        unit: str = "",
    ) -> None:
        self._gauge = PromGauge(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=get_metrics_registry(),
        )

    def labels(self, **labels: str) -> LabeledGauge:
        return LabeledGauge(self._gauge.labels(**labels))

    def set(self, value: float) -> None:
        self._gauge.set(value)

    def inc(self, amount: float = 1.0) -> None:
        self._gauge.inc(amount)

    def dec(self, amount: float = 1.0) -> None:
        self._gauge.dec(amount)


class LabeledGauge:
    """Gauge slice with labels applied."""

    def __init__(self, labeled: Any) -> None:
        self._labeled = labeled

    def set(self, value: float) -> None:
        self._labeled.set(value)

    def inc(self, amount: float = 1.0) -> None:
        self._labeled.inc(amount)

    def dec(self, amount: float = 1.0) -> None:
        self._labeled.dec(amount)
