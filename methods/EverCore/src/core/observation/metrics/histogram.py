"""
Histogram Wrapper

Provides a unified Histogram interface, isolating prometheus_client from business code.
"""
from prometheus_client import Histogram as PrometheusHistogram
from typing import Sequence
from .registry import get_metrics_registry


# Predefined bucket configurations
class HistogramBuckets:
    """Predefined Histogram bucket configurations"""
    
    # Default buckets (covering 5ms - 10s)
    DEFAULT = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    )
    
    # Fast operations (5ms - 500ms, for cache queries, simple calculations, etc.)
    FAST = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
    
    # API calls (10ms - 30s, for external API calls)
    # Denser buckets in 0.1-5s range for better P95/P99 accuracy
    API_CALL = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
    
    # Batch operations (100ms - 60s, for batch processing)
    BATCH = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
    
    # Embedding/Rerank (10ms - 10s, for ML inference)
    # Denser buckets in 0.1-3s range where most requests fall
    ML_INFERENCE = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    
    # Database queries (1ms - 5s)
    DATABASE = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class Histogram:
    """
    Histogram metric wrapper
    
    Features:
    - Distribution statistics of observed values
    - Suitable for latency, size, and other distribution data
    - Automatically calculates percentiles, mean, and sum
    
    Usage:
        from core.observation.metrics import Histogram, HistogramBuckets
        
        request_duration = Histogram(
            name='http_request_duration_seconds',
            description='HTTP request duration',
            labelnames=['method', 'path'],
            buckets=HistogramBuckets.API_CALL
        )
        
        # Usage
        request_duration.labels(method='GET', path='/api').observe(0.123)
        
        # Using context manager
        with request_duration.labels(method='GET', path='/api').time():
            do_something()
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = '',
        subsystem: str = '',
        unit: str = '',
        buckets: Sequence[float] = HistogramBuckets.DEFAULT,
    ):
        """
        Args:
            name: Metric name
            description: Metric description
            labelnames: List of label names
            namespace: Namespace (optional)
            subsystem: Subsystem (optional)
            unit: Unit (optional)
            buckets: Histogram bucket boundaries
        """
        registry = get_metrics_registry()
        
        self._histogram = PrometheusHistogram(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            buckets=buckets,
            registry=registry,
        )
        self._name = name
        self._labelnames = labelnames
    
    def labels(self, **labels) -> 'LabeledHistogram':
        """
        Return a Histogram with labels
        
        Returns:
            LabeledHistogram instance
        """
        labeled = self._histogram.labels(**labels)
        return LabeledHistogram(labeled)
    
    def observe(self, amount: float) -> None:
        """
        Record an observed value (no labels version)
        
        Args:
            amount: Observed value
        """
        self._histogram.observe(amount)
    
    def time(self):
        """
        Return a timing context manager (no labels version)
        
        Usage:
            with histogram.time():
                do_something()
        """
        return self._histogram.time()


class LabeledHistogram:
    """Histogram with labels"""
    
    def __init__(self, labeled_histogram):
        self._histogram = labeled_histogram
    
    def observe(self, amount: float) -> None:
        """
        Record an observed value
        
        Args:
            amount: Observed value
        """
        self._histogram.observe(amount)
    
    def time(self):
        """
        Return a timing context manager
        
        Usage:
            with histogram.labels(method='GET').time():
                do_something()
        """
        return self._histogram.time()

