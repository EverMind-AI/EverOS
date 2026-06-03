"""``Gauge`` / ``LabeledGauge`` — set / inc / dec; with & without labels."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from prometheus_client import CollectorRegistry

from everos.core.observability.metrics import (
    Gauge,
    reset_metrics_registry,
    set_metrics_registry,
)


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Swap in a fresh registry so test names don't clash with prod metrics."""
    set_metrics_registry(CollectorRegistry())
    yield
    reset_metrics_registry()


def _value(gauge: Gauge, **labels: str) -> float:
    """Read the gauge's current scalar value (helper for assertions)."""
    labeled = (
        gauge.labels(**labels)._labeled  # type: ignore[attr-defined]
        if labels
        else gauge._gauge  # type: ignore[attr-defined]
    )
    for sample in labeled.collect()[0].samples:
        if sample.name.endswith("_gauge") or "_" in sample.name:
            return float(sample.value)
    return float("nan")


def test_unlabeled_set_inc_dec() -> None:
    g = Gauge(name="queue_depth", description="rows pending")
    g.set(10)
    assert _value(g) == 10
    g.inc(2)
    assert _value(g) == 12
    g.dec()
    assert _value(g) == 11
    g.dec(5)
    assert _value(g) == 6


def test_labeled_isolates_streams() -> None:
    g = Gauge(name="cache_size", description="entries", labelnames=("region",))
    g.labels(region="us").set(100)
    g.labels(region="eu").set(50)
    g.labels(region="us").inc(5)
    g.labels(region="eu").dec(10)
    assert _value(g, region="us") == 105
    assert _value(g, region="eu") == 40


def test_namespace_subsystem_unit_render_in_metric_name() -> None:
    g = Gauge(
        name="depth",
        description="d",
        namespace="everos",
        subsystem="cascade",
        unit="rows",
    )
    g.set(7)
    # Underlying name should include all parts.
    full_name = g._gauge._name  # type: ignore[attr-defined]
    assert "everos" in full_name
    assert "cascade" in full_name
    assert "depth" in full_name
    assert "rows" in full_name
