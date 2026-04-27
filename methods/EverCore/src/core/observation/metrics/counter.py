"""
Counter Wrapper

Provides a unified Counter interface, isolating prometheus_client from business code.
"""
from prometheus_client import Counter as PrometheusCounter
from typing import Sequence
from .registry import get_metrics_registry


class Counter:
    """
    Counter metric wrapper
    
    Features:
    - Monotonically increasing counter (can only increment)
    - Suitable for total requests, total errors, etc.
    - Business code does not need to import prometheus_client directly
    
    Usage:
        from core.observation.metrics import Counter
        
        requests_total = Counter(
            name='http_requests_total',
            description='Total HTTP requests',
            labelnames=['method', 'path', 'status']
        )
        
        # 使用
        requests_total.labels(method='GET', path='/api', status='200').inc()
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = '',
        subsystem: str = '',
        unit: str = '',
    ):
        """
        Args:
            name: Metric name
            description: Metric description
            labelnames: List of label names
            namespace: Namespace (optional)
            subsystem: Subsystem (optional)
            unit: Unit (optional)
        """
        registry = get_metrics_registry()
        
        self._counter = PrometheusCounter(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=registry,
        )
        self._name = name
        self._labelnames = labelnames
    
    def labels(self, **labels) -> 'LabeledCounter':
        """
        Return a Counter with labels
        
        Returns:
            LabeledCounter instance
        """
        labeled = self._counter.labels(**labels)
        return LabeledCounter(labeled)
    
    def inc(self, amount: float = 1) -> None:
        """
        Increment counter (no labels version)
        
        Args:
            amount: Increment amount, defaults to 1
        """
        self._counter.inc(amount)


class LabeledCounter:
    """Counter with labels"""
    
    def __init__(self, labeled_counter):
        self._counter = labeled_counter
    
    def inc(self, amount: float = 1) -> None:
        """
        Increment counter
        
        Args:
            amount: Increment amount, defaults to 1
        """
        self._counter.inc(amount)

