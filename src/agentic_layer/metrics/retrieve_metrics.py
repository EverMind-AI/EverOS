"""
Retrieve Pipeline Metrics

Metrics for monitoring the complete memory retrieval pipeline including:
- Milvus vector search
- Memory fetch operations
- End-to-end retrieval latency

Usage:
    from agentic_layer.metrics import (
        RETRIEVE_REQUESTS_TOTAL,
        RETRIEVE_DURATION_SECONDS,
        RETRIEVE_RESULTS_COUNT,
        RETRIEVE_STAGE_DURATION_SECONDS,
        RETRIEVE_ERRORS_TOTAL,
    )
    
    # Record successful retrieval
    RETRIEVE_REQUESTS_TOTAL.labels(
        memory_type='episodic_memory',
        retrieve_method='vector',
        status='success'
    ).inc()
    
    # Record duration
    RETRIEVE_DURATION_SECONDS.labels(
        memory_type='episodic_memory',
        retrieve_method='vector_search'
    ).observe(0.567)
    
    # Record stage-specific duration
    RETRIEVE_STAGE_DURATION_SECONDS.labels(
        stage='milvus_search',
        memory_type='episodic_memory'
    ).observe(0.123)
"""

from core.observation.metrics import Counter, Histogram, HistogramBuckets


# ============================================================
# Counter Metrics
# ============================================================

RETRIEVE_REQUESTS_TOTAL = Counter(
    name='retrieve_requests_total',
    description='Total number of memory retrieval requests',
    labelnames=['memory_type', 'retrieve_method', 'status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memory retrieval requests counter

Labels:
- memory_type: episodic_memory, profile, foresight, event_log, entity, relation, etc.
- retrieve_method: vector, id_lookup, keyword, hybrid, rrf, agentic
- status: success, error, timeout, empty_result
"""


RETRIEVE_ERRORS_TOTAL = Counter(
    name='retrieve_errors_total',
    description='Total number of retrieval errors',
    labelnames=['retrieve_method', 'stage', 'error_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Retrieval errors counter

Labels:
- retrieve_method: keyword, vector, hybrid, rrf, agentic
- stage: keyword, vector, embedding, milvus_search, rerank, rrf_fusion
- error_type: connection_error, timeout, not_found, validation_error, unknown
"""


# ============================================================
# Histogram Metrics
# ============================================================

RETRIEVE_DURATION_SECONDS = Histogram(
    name='retrieve_duration_seconds',
    description='End-to-end duration of memory retrieval in seconds',
    labelnames=['memory_type', 'retrieve_method'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.API_CALL,  # 10ms - 30s for API calls
)
"""
End-to-end retrieval duration histogram

Labels:
- memory_type: episodic_memory, profile, foresight, event_log, etc.
- retrieve_method: vector, id_lookup, keyword, hybrid, rrf, agentic

Buckets: 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s
"""


RETRIEVE_STAGE_DURATION_SECONDS = Histogram(
    name='retrieve_stage_duration_seconds',
    description='Duration of individual retrieval stages in seconds',
    labelnames=['retrieve_method', 'stage', 'memory_type'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.DATABASE,  # 1ms - 5s for database operations
)
"""
Stage-specific duration histogram

Labels:
- retrieve_method: keyword, vector, hybrid, rrf, agentic
- stage: keyword, vector, embedding, milvus_search, rerank, rrf_fusion
- memory_type: episodic_memory, profile, foresight, event_log, etc.

Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s
"""


RETRIEVE_RESULTS_COUNT = Histogram(
    name='retrieve_results_count',
    description='Number of results returned from retrieval',
    labelnames=['memory_type', 'retrieve_method'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=(0, 1, 5, 10, 20, 50, 100, 200, 500, 1000),
)
"""
Retrieval results count histogram

Labels:
- memory_type: episodic_memory, profile, foresight, event_log, etc.
- retrieve_method: vector, id_lookup, keyword, hybrid, rrf, agentic

Buckets: 0, 1, 5, 10, 20, 50, 100, 200, 500, 1000 results
"""


# ============================================================
# Helper Functions
# ============================================================


def record_retrieve_request(
    memory_type: str,
    retrieve_method: str,
    status: str,
    duration_seconds: float,
    results_count: int,
) -> None:
    """
    Helper function to record all retrieval metrics in one call

    Args:
        memory_type: Memory type (episodic_memory, profile, foresight, etc.)
        retrieve_method: Retrieval method (keyword, vector, hybrid, rrf, agentic)
        status: Request status (success, error, timeout, empty_result)
        duration_seconds: Total retrieval duration in seconds
        results_count: Number of results returned

    Example:
        record_retrieve_request(
            memory_type='episodic_memory',
            retrieve_method='vector',
            status='success',
            duration_seconds=0.567,
            results_count=10
        )
    """
    # Counter
    RETRIEVE_REQUESTS_TOTAL.labels(
        memory_type=memory_type, retrieve_method=retrieve_method, status=status
    ).inc()

    # Duration histogram
    RETRIEVE_DURATION_SECONDS.labels(
        memory_type=memory_type, retrieve_method=retrieve_method
    ).observe(duration_seconds)

    # Results count histogram
    RETRIEVE_RESULTS_COUNT.labels(
        memory_type=memory_type, retrieve_method=retrieve_method
    ).observe(results_count)


def record_retrieve_stage(
    retrieve_method: str, stage: str, memory_type: str, duration_seconds: float
) -> None:
    """
    Helper function to record stage-specific duration

    Args:
        retrieve_method: Retrieval method (keyword, vector, hybrid, rrf, agentic)
        stage: Retrieval stage (keyword, vector, embedding, milvus_search, rerank, rrf_fusion)
        memory_type: Memory type
        duration_seconds: Stage duration in seconds

    Example:
        record_retrieve_stage(
            retrieve_method='vector',
            stage='milvus_search',
            memory_type='episodic_memory',
            duration_seconds=0.123
        )
    """
    RETRIEVE_STAGE_DURATION_SECONDS.labels(
        retrieve_method=retrieve_method, stage=stage, memory_type=memory_type
    ).observe(duration_seconds)


def record_retrieve_error(retrieve_method: str, stage: str, error_type: str) -> None:
    """
    Helper function to record retrieval error

    Args:
        retrieve_method: Retrieval method (keyword, vector, hybrid, rrf, agentic)
        stage: Stage where error occurred (keyword, vector, embedding, milvus_search, rerank, rrf_fusion)
        error_type: Error type (connection_error, timeout, not_found, validation_error, unknown)

    Example:
        record_retrieve_error(
            retrieve_method='vector',
            stage='milvus_search',
            error_type='timeout'
        )
    """
    RETRIEVE_ERRORS_TOTAL.labels(
        retrieve_method=retrieve_method, stage=stage, error_type=error_type
    ).inc()


class RetrieveMetricsContext:
    """
    Context manager for easy metrics recording in retrieval operations

    Usage:
        async def retrieve_memories(query, memory_type):
            with RetrieveMetricsContext(memory_type, 'vector_search') as ctx:
                # Stage 1: Embedding
                with ctx.stage('embedding'):
                    embedding = await get_embedding(query)

                # Stage 2: Milvus search
                with ctx.stage('milvus_search'):
                    results = await milvus_search(embedding)

                # Stage 3: Rerank
                with ctx.stage('rerank'):
                    reranked = await rerank(query, results)

                ctx.set_results_count(len(reranked))
                return reranked
    """

    def __init__(self, memory_type: str, retrieve_method: str):
        self.memory_type = memory_type
        self.retrieve_method = retrieve_method
        self.start_time = None
        self.results_count = 0
        self.status = 'success'
        self._current_stage = None
        self._stage_start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        if exc_type is not None:
            self.status = 'error'

        record_retrieve_request(
            memory_type=self.memory_type,
            retrieve_method=self.retrieve_method,
            status=self.status,
            duration_seconds=duration,
            results_count=self.results_count,
        )

        return False  # Don't suppress exceptions

    def stage(self, stage_name: str):
        """Context manager for stage timing"""
        return _StageContext(self, stage_name)

    def set_results_count(self, count: int):
        """Set the results count"""
        self.results_count = count

    def set_status(self, status: str):
        """Set the status (success, error, timeout, empty_result)"""
        self.status = status


class _StageContext:
    """Internal context manager for stage timing"""

    def __init__(self, parent: RetrieveMetricsContext, stage_name: str):
        self.parent = parent
        self.stage_name = stage_name
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        record_retrieve_stage(
            retrieve_method=self.parent.retrieve_method,
            stage=self.stage_name,
            memory_type=self.parent.memory_type,
            duration_seconds=duration,
        )

        if exc_type is not None:
            record_retrieve_error(
                retrieve_method=self.parent.retrieve_method,
                stage=self.stage_name,
                error_type='unknown' if exc_type is Exception else exc_type.__name__,
            )

        return False  # Don't suppress exceptions
