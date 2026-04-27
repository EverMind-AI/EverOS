"""
Metrics Library

Business code imports metric classes from here, no need to depend on prometheus_client directly.

Usage:
    from core.observation.metrics import Counter, Histogram, BaseGauge
    
    # Counter - monotonically increasing counter
    requests_total = Counter('http_requests_total', 'Total requests', ['method'])
    requests_total.labels(method='GET').inc()
    
    # Histogram - distribution statistics of observed values
    request_duration = Histogram(
        'http_request_duration_seconds', 
        'Request duration', 
        ['method'],
        buckets=HistogramBuckets.API_CALL
    )
    request_duration.labels(method='GET').observe(0.123)
    
    # BaseGauge - instantaneous value with auto-refresh (inheritance)
    class QueueSizeGauge(BaseGauge):
        def __init__(self, queue):
            super().__init__('queue_size', 'Queue size', ['queue_name'])
            self.queue = queue
        
        def refresh(self, labels: dict) -> float:
            return self.queue.qsize()
    
    # Using Gauge
    gauge = QueueSizeGauge(queue)
    gauge.labels(queue_name='main').start_refresh()  # default 5 second refresh
"""

from .counter import Counter, LabeledCounter
from .histogram import Histogram, LabeledHistogram, HistogramBuckets
from .gauge import BaseGauge, LabeledGauge
from .registry import (
    get_metrics_registry,
    set_metrics_registry,
    generate_metrics_response,
    reset_metrics_registry,
)
from .server import (
    start_metrics_server,
    is_metrics_server_running,
    get_metrics_url,
)

__all__ = [
    # Counter
    'Counter',
    'LabeledCounter',
    
    # Histogram
    'Histogram',
    'LabeledHistogram',
    'HistogramBuckets',
    
    # Gauge
    'BaseGauge',
    'LabeledGauge',
    
    # Registry
    'get_metrics_registry',
    'set_metrics_registry',
    'generate_metrics_response',
    'reset_metrics_registry',
    
    # Server
    'start_metrics_server',
    'is_metrics_server_running',
    'get_metrics_url',
]

