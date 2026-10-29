"""Prometheus-style metrics primitives + registry.

External usage:
    from everos.core.observability.metrics import (
        Counter, Gauge, Histogram, HistogramBuckets,
        configure_metrics, get_metrics_registry, generate_metrics_response,
    )
"""

from .counter import Counter as Counter
from .counter import LabeledCounter as LabeledCounter
from .gauge import Gauge as Gauge
from .gauge import LabeledGauge as LabeledGauge
from .histogram import Histogram as Histogram
from .histogram import HistogramBuckets as HistogramBuckets
from .histogram import LabeledHistogram as LabeledHistogram
from .registry import configure_metrics as configure_metrics
from .registry import generate_metrics_response as generate_metrics_response
from .registry import get_metrics_registry as get_metrics_registry

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "HistogramBuckets",
    "LabeledCounter",
    "LabeledGauge",
    "LabeledHistogram",
    "configure_metrics",
    "generate_metrics_response",
    "get_metrics_registry",
]
