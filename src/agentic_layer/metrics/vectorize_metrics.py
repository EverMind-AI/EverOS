"""
Vectorize (Embedding) Service Metrics

Metrics for monitoring embedding generation performance and reliability.

"""

from core.observation.metrics import Counter, Histogram, HistogramBuckets


# ============================================================
# Counter Metrics
# ============================================================

VECTORIZE_REQUESTS_TOTAL = Counter(
    name='vectorize_requests_total',
    description='Total number of vectorize (embedding) requests',
    labelnames=['provider', 'operation', 'status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Vectorize requests counter

Labels:
- provider: vllm, deepinfra
- operation: get_embedding, get_embeddings, get_embeddings_batch
- status: success, error, timeout, fallback
"""


VECTORIZE_FALLBACK_TOTAL = Counter(
    name='vectorize_fallback_total',
    description='Total number of vectorize fallback events',
    labelnames=['primary_provider', 'fallback_provider', 'reason'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Vectorize fallback counter

Labels:
- primary_provider: Primary provider that failed (vllm, deepinfra)
- fallback_provider: Fallback provider used (vllm, deepinfra)
- reason: error, timeout, max_failures_exceeded
"""


VECTORIZE_ERRORS_TOTAL = Counter(
    name='vectorize_errors_total',
    description='Total number of vectorize errors',
    labelnames=['provider', 'operation', 'error_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Vectorize errors counter

Labels:
- provider: vllm, deepinfra
- operation: get_embedding, get_embeddings, get_embeddings_batch
- error_type: api_error, timeout, rate_limit, validation_error, unknown
"""


VECTORIZE_TOKENS_TOTAL = Counter(
    name='vectorize_tokens_total',
    description='Total number of tokens processed for embedding',
    labelnames=['provider'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Vectorize tokens counter (for cost tracking)

Labels:
- provider: vllm, deepinfra
"""


# ============================================================
# Histogram Metrics
# ============================================================

VECTORIZE_DURATION_SECONDS = Histogram(
    name='vectorize_duration_seconds',
    description='Duration of vectorize operations in seconds',
    labelnames=['provider', 'operation'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.ML_INFERENCE,  # 10ms - 10s for ML inference
)
"""
Vectorize operation duration histogram

Labels:
- provider: vllm, deepinfra
- operation: get_embedding, get_embeddings, get_embeddings_batch

Buckets: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
"""


VECTORIZE_BATCH_SIZE = Histogram(
    name='vectorize_batch_size',
    description='Batch size of vectorize operations',
    labelnames=['provider', 'operation'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)
"""
Vectorize batch size histogram

Labels:
- provider: vllm, deepinfra
- operation: get_embeddings, get_embeddings_batch

Buckets: 1, 2, 5, 10, 20, 50, 100, 200, 500 texts
"""


# ============================================================
# Helper Functions
# ============================================================

def record_vectorize_request(
    provider: str,
    operation: str,
    status: str,
    duration_seconds: float,
    batch_size: int = 1,
    tokens: int = 0,
) -> None:
    """
    Helper function to record all vectorize metrics in one call
    
    Args:
        provider: Service provider (vllm, deepinfra)
        operation: Operation type (get_embedding, get_embeddings, get_embeddings_batch)
        status: Request status (success, error, timeout, fallback)
        duration_seconds: Operation duration in seconds
        batch_size: Number of texts processed
        tokens: Number of tokens processed (optional, for cost tracking)
    
    Example:
        record_vectorize_request(
            provider='vllm',
            operation='get_embeddings',
            status='success',
            duration_seconds=0.5,
            batch_size=10,
            tokens=250
        )
    """
    # Counter
    VECTORIZE_REQUESTS_TOTAL.labels(
        provider=provider,
        operation=operation,
        status=status
    ).inc()
    
    # Duration histogram
    VECTORIZE_DURATION_SECONDS.labels(
        provider=provider,
        operation=operation
    ).observe(duration_seconds)
    
    # Batch size histogram (only for batch operations)
    if batch_size > 1:
        VECTORIZE_BATCH_SIZE.labels(
            provider=provider,
            operation=operation
        ).observe(batch_size)
    
    # Token counter (if available)
    if tokens > 0:
        VECTORIZE_TOKENS_TOTAL.labels(provider=provider).inc(tokens)


def record_vectorize_fallback(
    primary_provider: str,
    fallback_provider: str,
    reason: str,
) -> None:
    """
    Helper function to record vectorize fallback event
    
    Args:
        primary_provider: Primary provider that failed
        fallback_provider: Fallback provider used
        reason: Fallback reason (error, timeout, max_failures_exceeded)
    
    Example:
        record_vectorize_fallback(
            primary_provider='vllm',
            fallback_provider='deepinfra',
            reason='timeout'
        )
    """
    VECTORIZE_FALLBACK_TOTAL.labels(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        reason=reason
    ).inc()


def record_vectorize_error(
    provider: str,
    operation: str,
    error_type: str,
) -> None:
    """
    Helper function to record vectorize error
    
    Args:
        provider: Service provider
        operation: Operation type
        error_type: Error type (api_error, timeout, rate_limit, validation_error, unknown)
    
    Example:
        record_vectorize_error(
            provider='vllm',
            operation='get_embedding',
            error_type='timeout'
        )
    """
    VECTORIZE_ERRORS_TOTAL.labels(
        provider=provider,
        operation=operation,
        error_type=error_type
    ).inc()

