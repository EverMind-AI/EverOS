"""Tracing — W3C id generation + OpenTelemetry tracer lifecycle.

External usage::

    from everos.core.observability.tracing import (
        gen_request_id,
        get_tracer,
        init_tracing,
        shutdown_tracing,
        force_flush,
    )

``get_tracer`` is safe to call unconditionally: it returns a no-op tracer
until ``init_tracing`` installs a provider (and when the optional ``[otel]``
extra is not installed), so call sites never need to branch on config.
"""

from __future__ import annotations

from .attributes import capture_input as capture_input
from .attributes import capture_output as capture_output
from .attributes import current_trace_ids as current_trace_ids
from .attributes import current_traceparent as current_traceparent
from .attributes import memory_span as memory_span
from .attributes import set_capture_content as set_capture_content
from .attributes import set_generation_usage as set_generation_usage
from .attributes import set_redactor as set_redactor
from .attributes import use_traceparent as use_traceparent
from .ids import gen_request_id as gen_request_id
from .provider import force_flush as force_flush
from .provider import get_tracer as get_tracer
from .provider import init_tracing as init_tracing
from .provider import shutdown_tracing as shutdown_tracing
from .scores import emit_recall_scores as emit_recall_scores
from .scores import init_score_sink as init_score_sink
from .scores import shutdown_score_sink as shutdown_score_sink

__all__ = [
    "capture_input",
    "capture_output",
    "current_trace_ids",
    "current_traceparent",
    "emit_recall_scores",
    "force_flush",
    "gen_request_id",
    "get_tracer",
    "init_score_sink",
    "init_tracing",
    "memory_span",
    "set_capture_content",
    "set_generation_usage",
    "set_redactor",
    "shutdown_score_sink",
    "shutdown_tracing",
    "use_traceparent",
]
