---
paths:
  - "src/**/*.py"
---

# Logging & observability rule

- **Use the project logger**, never `print` or the stdlib `logging` directly:
  ```python
  from everos.core.observability.logging import get_logger
  logger = get_logger(__name__)
  ```
- **Structured logging** (`structlog`): pass context as keyword fields, not f-strings.
  ```python
  logger.info("memory.search.completed", owner_type=owner, n_results=len(items))
  ```
  Event name first (dotted, stable), structured kwargs after. This keeps logs
  queryable and avoids leaking interpolated PII into the message string.
- **Levels**: `debug` for developer detail, `info` for lifecycle milestones,
  `warning` for recoverable anomalies, `error` for failures with a stack/context.
- **Metrics** go through `core.observability.metrics` (Prometheus); don't invent
  ad-hoc counters. Histograms/counters/gauges have registry helpers.
- Don't log secrets, API keys, or full memory content at `info`/above.
- **Tracing** (optional, `[otel]` extra, **off by default**): open spans with
  `memory_span(...)` from `core.observability.tracing` — it stamps the Langfuse
  `langfuse.*` attributes and is a no-op until `[observability] enabled`, so call
  sites never branch on config. LLM / embedding token usage rides
  `set_generation_usage` onto the active span (Langfuse computes cost).
  Request/response content is emitted only when `capture_content` is on
  (redaction hook + truncation). `request_id` is kept independent of the OTel
  `trace_id`; an upstream `traceparent` header is continued when present.
