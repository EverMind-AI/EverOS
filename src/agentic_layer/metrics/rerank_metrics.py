"""
Rerank Service Metrics

Metrics for monitoring reranking service performance and reliability.

Usage:
    from agentic_layer.metrics import (
        RERANK_REQUESTS_TOTAL,
        RERANK_DURATION_SECONDS,
        RERANK_DOCUMENTS_TOTAL,
        RERANK_FALLBACK_TOTAL,
        RERANK_ERRORS_TOTAL,
    )
    
    # Record successful rerank request
    RERANK_REQUESTS_TOTAL.labels(
        provider='vllm',
        status='success'
    ).inc()
    
    # Record duration
    RERANK_DURATION_SECONDS.labels(
        provider='vllm'
    ).observe(0.234)
    
    # Record documents count
    RERANK_DOCUMENTS_TOTAL.labels(
        provider='vllm'
    ).observe(50)
"""

from core.observation.metrics import Counter, Histogram, HistogramBuckets


# ============================================================
# Counter Metrics
# ============================================================

RERANK_REQUESTS_TOTAL = Counter(
    name='rerank_requests_total',
    description='Total number of rerank requests',
    labelnames=['provider', 'status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Rerank requests counter

Labels:
- provider: vllm, deepinfra
- status: success, error, timeout, fallback
"""


RERANK_FALLBACK_TOTAL = Counter(
    name='rerank_fallback_total',
    description='Total number of rerank fallback events',
    labelnames=['primary_provider', 'fallback_provider', 'reason'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Rerank fallback counter

Labels:
- primary_provider: Primary provider that failed (vllm, deepinfra)
- fallback_provider: Fallback provider used (vllm, deepinfra)
- reason: error, timeout, max_failures_exceeded
"""


RERANK_ERRORS_TOTAL = Counter(
    name='rerank_errors_total',
    description='Total number of rerank errors',
    labelnames=['provider', 'error_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Rerank errors counter

Labels:
- provider: vllm, deepinfra
- error_type: api_error, timeout, rate_limit, validation_error, unknown
"""


# ============================================================
# Histogram Metrics
# ============================================================

RERANK_DURATION_SECONDS = Histogram(
    name='rerank_duration_seconds',
    description='Duration of rerank operations in seconds',
    labelnames=['provider'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.ML_INFERENCE,  # 10ms - 10s for ML inference
)
"""
Rerank operation duration histogram

Labels:
- provider: vllm, deepinfra

Buckets: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
"""


RERANK_DOCUMENTS_TOTAL = Histogram(
    name='rerank_documents_count',
    description='Number of documents reranked per request',
    labelnames=['provider'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000),
)
"""
Rerank documents count histogram

Labels:
- provider: vllm, deepinfra

Buckets: 1, 5, 10, 20, 50, 100, 200, 500, 1000 documents
"""


# ============================================================
# Helper Functions
# ============================================================

def record_rerank_request(
    provider: str,
    status: str,
    duration_seconds: float,
    documents_count: int,
) -> None:
    """
    Helper function to record all rerank metrics in one call
    
    Args:
        provider: Service provider (vllm, deepinfra)
        status: Request status (success, error, timeout, fallback)
        duration_seconds: Operation duration in seconds
        documents_count: Number of documents reranked
    
    Example:
        record_rerank_request(
            provider='vllm',
            status='success',
            duration_seconds=0.5,
            documents_count=50
        )
    """
    # Counter
    RERANK_REQUESTS_TOTAL.labels(
        provider=provider,
        status=status
    ).inc()
    
    # Duration histogram
    RERANK_DURATION_SECONDS.labels(
        provider=provider
    ).observe(duration_seconds)
    
    # Documents count histogram
    RERANK_DOCUMENTS_TOTAL.labels(
        provider=provider
    ).observe(documents_count)


def record_rerank_fallback(
    primary_provider: str,
    fallback_provider: str,
    reason: str,
) -> None:
    """
    Helper function to record rerank fallback event
    
    Args:
        primary_provider: Primary provider that failed
        fallback_provider: Fallback provider used
        reason: Fallback reason (error, timeout, max_failures_exceeded)
    
    Example:
        record_rerank_fallback(
            primary_provider='vllm',
            fallback_provider='deepinfra',
            reason='timeout'
        )
    """
    RERANK_FALLBACK_TOTAL.labels(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        reason=reason
    ).inc()


def record_rerank_error(
    provider: str,
    error_type: str,
) -> None:
    """
    Helper function to record rerank error
    
    Args:
        provider: Service provider
        error_type: Error type (api_error, timeout, rate_limit, validation_error, unknown)
    
    Example:
        record_rerank_error(
            provider='vllm',
            error_type='timeout'
        )
    """
    RERANK_ERRORS_TOTAL.labels(
        provider=provider,
        error_type=error_type
    ).inc()
