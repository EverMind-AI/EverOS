"""Span helpers implementing the Langfuse / OpenTelemetry attribute contract.

``memory_span`` opens a span under the shared ``everos`` tracer and stamps
the ``langfuse.*`` trace/observation attributes (observation type, session /
user ids, trace metadata, tags). ``set_generation_usage`` writes the
``gen_ai.*`` model + token attributes onto the *current* span, so an LLM
client wrapper can record usage without knowing which span is active.

Request/response content (``langfuse.observation.input/output``) is
privacy-gated: ``capture_input`` / ``capture_output`` only emit it when
``capture_content`` is on, after a redaction hook + truncation. Off by
default, so spans carry metadata only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .provider import get_tracer

# ── langfuse.* trace/observation attribute keys ──────────────────────────
LF_OBSERVATION_TYPE = "langfuse.observation.type"
LF_SESSION_ID = "langfuse.session.id"
LF_USER_ID = "langfuse.user.id"
LF_TAGS = "langfuse.trace.tags"
LF_METADATA_PREFIX = "langfuse.trace.metadata."

# ── gen_ai.* generation attribute keys (Langfuse computes cost from these) ─
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# ── content capture (privacy-gated) ──────────────────────────────────────
LF_OBSERVATION_INPUT = "langfuse.observation.input"
LF_OBSERVATION_OUTPUT = "langfuse.observation.output"
_MAX_CONTENT_CHARS = 4096

DEFAULT_TAGS: tuple[str, ...] = ("everos", "memory")

# Off by default: no request/response content leaves the process unless
# capture_content is turned on (set once at init_tracing from settings).
_capture_content = False
_redactor: Callable[[str], str] = lambda text: text  # noqa: E731 - overridable hook


def set_capture_content(enabled: bool) -> None:
    """Toggle content capture (called from init_tracing / shutdown)."""
    global _capture_content
    _capture_content = enabled


def set_redactor(redactor: Callable[[str], str] | None) -> None:
    """Install a redaction hook applied to captured content; None resets it."""
    global _redactor
    _redactor = redactor if redactor is not None else (lambda text: text)


def _prepare_content(value: Any) -> str:
    """Serialize, redact, then truncate content for a span attribute."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = _redactor(text)
    return text[:_MAX_CONTENT_CHARS]


def capture_input(span: Any, value: Any) -> None:
    """Set ``langfuse.observation.input`` — only when capture_content is on."""
    if _capture_content and value is not None:
        span.set_attribute(LF_OBSERVATION_INPUT, _prepare_content(value))


def capture_output(span: Any, value: Any) -> None:
    """Set ``langfuse.observation.output`` — only when capture_content is on."""
    if _capture_content and value is not None:
        span.set_attribute(LF_OBSERVATION_OUTPUT, _prepare_content(value))


try:
    from opentelemetry import trace as _otel_trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - only without the [otel] extra
    _OTEL_AVAILABLE = False


def _has_active_span() -> bool:
    """True when a valid span context is currently active (a parent exists)."""
    if not _OTEL_AVAILABLE:
        return False
    return _otel_trace.get_current_span().get_span_context().is_valid


@contextmanager
def memory_span(
    name: str,
    *,
    observation_type: str,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] = DEFAULT_TAGS,
    nested_only: bool = False,
) -> Iterator[Any]:
    """Open a span named ``name`` and stamp the langfuse.* attributes.

    Args:
        name: Span name (e.g. ``everos.memory.search``).
        observation_type: ``langfuse.observation.type`` — span / generation /
            embedding / retriever / agent.
        session_id / user_id: Grouping ids (dropped when None).
        metadata: Flat mapping → ``langfuse.trace.metadata.<key>``; None
            values are dropped rather than emitted as the string "None".
        tags: ``langfuse.trace.tags`` list.
        nested_only: When True, only open a span if one is already active.
            Calls that run outside any request trace (e.g. cascade-time
            embedding during indexing) would otherwise each start a NEW root
            trace — a per-chunk trace explosion — so they no-op instead.
    """
    if nested_only and not _has_active_span():
        yield _otel_trace.get_current_span() if _OTEL_AVAILABLE else None
        return
    tracer = get_tracer("everos")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(LF_OBSERVATION_TYPE, observation_type)
        if session_id:
            span.set_attribute(LF_SESSION_ID, session_id)
        if user_id:
            span.set_attribute(LF_USER_ID, user_id)
        if tags:
            span.set_attribute(LF_TAGS, list(tags))
        for key, value in (metadata or {}).items():
            if value is not None:
                span.set_attribute(f"{LF_METADATA_PREFIX}{key}", value)
        yield span


def current_traceparent() -> str | None:
    """W3C ``traceparent`` for the current span, or None when there is none.

    Captured where a request's span is active (e.g. OME enqueue) and carried
    across the async / process boundary so a background strategy span can
    re-attach to the originating trace.
    """
    if not _OTEL_AVAILABLE:
        return None
    from opentelemetry.propagate import inject

    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


@contextmanager
def use_traceparent(traceparent: str | None) -> Iterator[None]:
    """Attach ``traceparent`` as the current context for the block.

    Spans opened inside become children of that (remote) trace. No-op when
    the traceparent is absent or OTel is not installed — the span then roots
    its own trace.
    """
    if not _OTEL_AVAILABLE or not traceparent:
        yield
        return
    from opentelemetry import context as otel_context
    from opentelemetry.propagate import extract

    token = otel_context.attach(extract({"traceparent": traceparent}))
    try:
        yield
    finally:
        otel_context.detach(token)


def current_trace_ids() -> tuple[str, str] | None:
    """Return ``(trace_id_hex_032x, span_id_hex_016x)`` of the current span.

    Returns None when OTel is absent or there is no valid recording span —
    the exact hex mapping Langfuse's OTLP ingestion uses for traceId /
    observationId, so recall scores attach to the right observation.
    """
    if not _OTEL_AVAILABLE:
        return None
    ctx = _otel_trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def set_generation_usage(
    *,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Record ``gen_ai.*`` model + token attributes on the current span.

    Token counts ACCUMULATE: a span that wraps more than one ``chat`` call
    (e.g. the agentic/rank search path issues several LLM calls under one
    ``everos.search.rank`` span) sums each call's usage rather than letting
    the last call overwrite the rest. ``model`` is set as-is (last wins).

    No-op when OTel is absent or there is no active recording span, so LLM
    client wrappers can call it unconditionally.
    """
    if not _OTEL_AVAILABLE:
        return
    span = _otel_trace.get_current_span()
    # `.attributes` exists on a recording SDK span and reflects values set
    # earlier in this span's life; a non-recording span has none (getattr
    # falls back to {}), and set_attribute on it is itself a no-op.
    existing = getattr(span, "attributes", None) or {}
    if model is not None:
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
    if input_tokens is not None:
        prior = existing.get(GEN_AI_USAGE_INPUT_TOKENS, 0)
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, prior + input_tokens)
    if output_tokens is not None:
        prior = existing.get(GEN_AI_USAGE_OUTPUT_TOKENS, 0)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, prior + output_tokens)
