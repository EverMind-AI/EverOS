"""
LLM Provider Metrics

Prometheus metrics for monitoring LLM API call volume and error rates.
Co-located with the LLM provider per Prometheus instrumentation best practices.
"""

from core.observation.metrics import Counter


# ============================================================
# Counter Metrics
# ============================================================

LLM_REQUESTS_TOTAL = Counter(
    name='llm_requests_total',
    description='Total number of LLM API requests',
    labelnames=['model', 'status'],
    namespace='evermemos',
    subsystem='memory_layer',
)
"""
LLM requests counter.

Labels:
- model: LLM model name (e.g. "gpt-4.1-mini", "qwen/qwen3-235b-a22b-2507")
- status: Request outcome
    - success: HTTP 200 with valid response
    - rate_limit: HTTP 429 (all keys exhausted)
    - key_error: HTTP 401/402/403 (all keys exhausted)
    - server_error: HTTP 5xx (after max retries)
    - client_error: Network / connection error (after max retries)
    - request_error: HTTP 400/404/422 (no retry)

PromQL examples:
    # Total requests per second
    rate(evermemos_memory_layer_llm_requests_total[5m])

    # 429 count
    evermemos_memory_layer_llm_requests_total{status="rate_limit"}

    # 429 ratio
    evermemos_memory_layer_llm_requests_total{status="rate_limit"}
      / evermemos_memory_layer_llm_requests_total
"""


# ============================================================
# Helper Functions
# ============================================================


def record_llm_request(model: str, status: str) -> None:
    """Record an LLM request outcome.

    Args:
        model: LLM model name.
        status: Request outcome (success, rate_limit, key_error,
                server_error, client_error, request_error).
    """
    LLM_REQUESTS_TOTAL.labels(model=model, status=status).inc()
