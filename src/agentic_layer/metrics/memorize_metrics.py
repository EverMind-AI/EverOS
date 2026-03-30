"""
Memorize Pipeline Metrics

Metrics for monitoring the memory ingestion (add memory) pipeline including:
- Request processing
- Memory extraction statistics
- Boundary detection and MemCell extraction

All metrics include space_id and raw_data_type labels for multi-tenant support.

Usage:
    from agentic_layer.metrics.memorize_metrics import (
        record_memorize_request,
        record_memorize_error,
        record_extraction_stage,
        get_space_id_for_metrics,
    )
    
    # Record successful memorize request
    record_memorize_request(
        space_id=get_space_id_for_metrics(),
        raw_data_type='conversation',
        status='success',
        duration_seconds=0.5,
    )
    
    # Record extraction stage duration
    record_extraction_stage(
        space_id=get_space_id_for_metrics(),
        raw_data_type='conversation',
        stage='extract_episodes',
        duration_seconds=2.5,
    )
"""

from typing import Optional
from core.observation.metrics import Counter, Histogram, HistogramBuckets
from core.tenants.tenant_contextvar import get_current_tenant


# ============================================================
# Utility Functions
# ============================================================

def get_space_id_for_metrics() -> str:
    """
    Get space_id for metrics label
    
    Returns 'default' if tenant context is not available or space_id is not set.
    
    Returns:
        str: space_id or 'default'
    """
    try:
        tenant = get_current_tenant()
        if tenant and tenant.tenant_detail and tenant.tenant_detail.tenant_info:
            return tenant.tenant_detail.tenant_info.get('space_id', 'default')
    except Exception:
        pass
    return 'default'


def get_raw_data_type_label(raw_data_type: Optional[str]) -> str:
    """
    Get raw_data_type for metrics label
    
    Args:
        raw_data_type: Raw data type string or enum value
        
    Returns:
        str: Original value as string or 'unknown'
    """
    if not raw_data_type:
        return 'unknown'
    return str(raw_data_type)


# ============================================================
# Counter Metrics
# ============================================================

MEMORIZE_REQUESTS_TOTAL = Counter(
    name='memorize_requests_total',
    description='Total number of memorize requests',
    labelnames=['space_id', 'raw_data_type', 'status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memorize requests counter

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- status: success, error, accumulated, extracted
  - success: Request processed successfully (with or without memory extraction)
  - error: Request failed
  - accumulated: No memory extracted, message queued
  - extracted: Memories extracted successfully
"""


MEMORIZE_ERRORS_TOTAL = Counter(
    name='memorize_errors_total',
    description='Total number of memorize errors',
    labelnames=['space_id', 'raw_data_type', 'stage', 'error_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memorize errors counter

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- stage: conversion, save_logs, memorize_process
- error_type: validation_error, timeout, connection_error, unknown
"""


# ============================================================
# Histogram Metrics
# ============================================================

MEMORIZE_DURATION_SECONDS = Histogram(
    name='memorize_duration_seconds',
    description='End-to-end duration of memorize operation in seconds',
    labelnames=['space_id', 'raw_data_type', 'status'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.API_CALL,  # 10ms - 30s for API calls
)
"""
End-to-end memorize duration histogram

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- status: success, error

Buckets: 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s
"""


MEMORIZE_MESSAGES_TOTAL = Counter(
    name='memorize_messages_total',
    description='Total number of messages processed for memorization',
    labelnames=['space_id', 'raw_data_type', 'status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Messages processed counter

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- status: received, saved, processed
"""


# ============================================================
# Boundary Detection Metrics
# ============================================================

BOUNDARY_DETECTION_TOTAL = Counter(
    name='boundary_detection_total',
    description='Total number of boundary detection results',
    labelnames=['space_id', 'raw_data_type', 'result', 'trigger_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Boundary detection results counter

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- result: should_end, should_wait, error, force_split
- trigger_type: llm, token_limit, message_limit, first_message
"""


MEMCELL_EXTRACTED_TOTAL = Counter(
    name='memcell_extracted_total',
    description='Total number of MemCells extracted',
    labelnames=['space_id', 'raw_data_type', 'trigger_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
MemCell extraction counter

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- trigger_type: llm, token_limit, message_limit
"""


# ============================================================
# Memory Extraction Metrics
# ============================================================

MEMORY_EXTRACTION_STAGE_DURATION_SECONDS = Histogram(
    name='memory_extraction_stage_duration_seconds',
    description='Duration of individual memory extraction stages in seconds',
    labelnames=['space_id', 'raw_data_type', 'stage'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.ML_INFERENCE,  # LLM inference buckets
)
"""
Memory extraction stage duration histogram

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- stage: init_state, extract_episodes, extract_foresights, extract_event_logs, 
         update_memcell_cluster, process_memories

Buckets: 10ms - 10s for ML inference
"""


MEMORY_EXTRACTED_TOTAL = Counter(
    name='memory_extracted_total',
    description='Total number of memories extracted by type',
    labelnames=['space_id', 'raw_data_type', 'memory_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memory extraction counter by type

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- memory_type: episode, foresight, event_log
"""


EXTRACT_MEMORY_REQUESTS_TOTAL = Counter(
    name='extract_memory_requests_total',
    description='Total number of extract_memory calls by memory type',
    labelnames=['space_id', 'raw_data_type', 'memory_type', 'status'],
    namespace='evermemos',
    subsystem='memory_layer',
)
"""
extract_memory call counter

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- memory_type: episodic_memory, foresight, event_log, profile, group_profile
- status: success, error, empty_result
"""


EXTRACT_MEMORY_DURATION_SECONDS = Histogram(
    name='extract_memory_duration_seconds',
    description='Duration of extract_memory calls by memory type in seconds',
    labelnames=['space_id', 'raw_data_type', 'memory_type'],
    namespace='evermemos',
    subsystem='memory_layer',
    buckets=HistogramBuckets.ML_INFERENCE,  # LLM inference buckets
)
"""
extract_memory duration histogram

Labels:
- space_id: Tenant space identifier
- raw_data_type: Type of raw data (conversation, etc.)
- memory_type: episodic_memory, foresight, event_log, profile, group_profile

Buckets: 10ms - 10s for ML inference
"""


# ============================================================
# Helper Functions
# ============================================================

def record_memorize_request(
    space_id: str,
    raw_data_type: str,
    status: str,
    duration_seconds: float,
) -> None:
    """
    Helper function to record memorize request metrics
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        status: Request status (success, error, accumulated, extracted)
        duration_seconds: Total operation duration in seconds
    
    Example:
        record_memorize_request(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            status='extracted',
            duration_seconds=0.5,
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    
    # Counter
    MEMORIZE_REQUESTS_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, status=status
    ).inc()
    
    # Duration histogram (use simplified status for duration)
    duration_status = 'success' if status in ('success', 'accumulated', 'extracted') else 'error'
    MEMORIZE_DURATION_SECONDS.labels(
        space_id=space_id, raw_data_type=raw_data_type, status=duration_status
    ).observe(duration_seconds)


def record_memorize_error(
    space_id: str,
    raw_data_type: str,
    stage: str,
    error_type: str,
) -> None:
    """
    Helper function to record memorize error
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        stage: Stage where error occurred (conversion, save_logs, memorize_process)
        error_type: Error type (validation_error, timeout, connection_error, unknown)
    
    Example:
        record_memorize_error(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            stage='conversion',
            error_type='validation_error',
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    MEMORIZE_ERRORS_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, stage=stage, error_type=error_type
    ).inc()


def record_memorize_message(
    space_id: str,
    raw_data_type: str,
    status: str,
    count: int = 1,
) -> None:
    """
    Helper function to record message processing
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        status: Message status (received, saved, processed)
        count: Number of messages
    
    Example:
        record_memorize_message(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            status='received',
            count=1,
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    MEMORIZE_MESSAGES_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, status=status
    ).inc(count)


def classify_memorize_error(error: Exception) -> str:
    """
    Classify error type for metrics
    
    Args:
        error: Exception instance
    
    Returns:
        Error type string for metrics label
    """
    # TODO: Add detailed error classification based on actual scenarios
    _ = error  # Placeholder for future use
    return 'error'


def record_boundary_detection(
    space_id: str,
    raw_data_type: str,
    result: str,
    trigger_type: str,
) -> None:
    """
    Helper function to record boundary detection metrics
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        result: Detection result (should_end, should_wait, error, force_split)
        trigger_type: What triggered the detection (llm, token_limit, message_limit, first_message)
    
    Example:
        record_boundary_detection(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            result='should_end',
            trigger_type='llm',
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    BOUNDARY_DETECTION_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, result=result, trigger_type=trigger_type
    ).inc()


def record_memcell_extracted(
    space_id: str,
    raw_data_type: str,
    trigger_type: str,
) -> None:
    """
    Helper function to record MemCell extraction
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        trigger_type: What triggered the extraction (llm, token_limit, message_limit)
    
    Example:
        record_memcell_extracted(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            trigger_type='llm',
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    MEMCELL_EXTRACTED_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, trigger_type=trigger_type
    ).inc()


def record_extraction_stage(
    space_id: str,
    raw_data_type: str,
    stage: str,
    duration_seconds: float,
) -> None:
    """
    Helper function to record memory extraction stage duration
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        stage: Extraction stage (init_state, extract_episodes, extract_foresights, 
               extract_event_logs, update_memcell_cluster, process_memories)
        duration_seconds: Stage duration in seconds
    
    Example:
        record_extraction_stage(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            stage='extract_episodes',
            duration_seconds=2.5,
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    MEMORY_EXTRACTION_STAGE_DURATION_SECONDS.labels(
        space_id=space_id, raw_data_type=raw_data_type, stage=stage
    ).observe(duration_seconds)


def record_memory_extracted(
    space_id: str,
    raw_data_type: str,
    memory_type: str,
    count: int = 1,
) -> None:
    """
    Helper function to record extracted memory count by type
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        memory_type: Memory type (episode, foresight, event_log)
        count: Number of memories extracted
    
    Example:
        record_memory_extracted(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            memory_type='episode',
            count=3,
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    MEMORY_EXTRACTED_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, memory_type=memory_type
    ).inc(count)


def record_extract_memory_call(
    space_id: str,
    raw_data_type: str,
    memory_type: str,
    status: str,
    duration_seconds: float,
) -> None:
    """
    Helper function to record extract_memory call metrics
    
    Args:
        space_id: Tenant space identifier
        raw_data_type: Type of raw data (conversation, etc.)
        memory_type: Memory type (episodic_memory, foresight, event_log, profile, group_profile)
        status: Call status (success, error, empty_result)
        duration_seconds: Call duration in seconds
    
    Example:
        record_extract_memory_call(
            space_id=get_space_id_for_metrics(),
            raw_data_type='conversation',
            memory_type='episodic_memory',
            status='success',
            duration_seconds=2.5,
        )
    """
    raw_data_type = get_raw_data_type_label(raw_data_type)
    EXTRACT_MEMORY_REQUESTS_TOTAL.labels(
        space_id=space_id, raw_data_type=raw_data_type, memory_type=memory_type, status=status
    ).inc()
    EXTRACT_MEMORY_DURATION_SECONDS.labels(
        space_id=space_id, raw_data_type=raw_data_type, memory_type=memory_type
    ).observe(duration_seconds)
