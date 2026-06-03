"""Histogram wrapper around ``prometheus_client.Histogram``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prometheus_client import Histogram as PromHistogram

from .registry import get_metrics_registry


class HistogramBuckets:
    """Predefined bucket configurations for common workloads."""

    DEFAULT: tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    )
    FAST: tuple[float, ...] = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
    API_CALL: tuple[float, ...] = (
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        30.0,
    )
    BATCH: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
    DATABASE: tuple[float, ...] = (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    )


class Histogram:
    """Distribution of observed values (latency, sizes)."""

    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = "",
        subsystem: str = "",
        unit: str = "",
        buckets: Sequence[float] = HistogramBuckets.DEFAULT,
    ) -> None:
        self._histogram = PromHistogram(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            buckets=tuple(buckets),
            registry=get_metrics_registry(),
        )

    def labels(self, **labels: str) -> LabeledHistogram:
        return LabeledHistogram(self._histogram.labels(**labels))

    def observe(self, amount: float) -> None:
        self._histogram.observe(amount)

    def time(self) -> Any:
        return self._histogram.time()


class LabeledHistogram:
    """Histogram slice with labels applied."""

    def __init__(self, labeled: Any) -> None:
        self._labeled = labeled

    def observe(self, amount: float) -> None:
        self._labeled.observe(amount)

    def time(self) -> Any:
        return self._labeled.time()
