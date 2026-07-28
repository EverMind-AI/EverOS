"""OpenTelemetry tracer provider lifecycle + a no-op-safe tracer facade.

OpenTelemetry is an optional dependency (the ``[otel]`` extra). This module
never fails to import when it is absent: the SDK imports are guarded, and
``get_tracer`` returns a no-op tracer until ``init_tracing`` installs a real
provider.

The provider is held here (module-local) rather than on the OTel *global*
so it can be built and torn down repeatedly — in tests and across restarts —
without tripping OTel's "set global provider once" guard. Span context
propagation (parent/child nesting) still works: that rides OTel's context
vars, which are independent of which provider produced the tracer.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - only without the [otel] extra
    _OTEL_AVAILABLE = False

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanProcessor

    from everos.config.settings import ObservabilitySettings

# Our TracerProvider, deliberately kept off the OTel global (see module docstring).
_provider: Any = None


class _NoopSpan:
    """Span stand-in used when tracing is off or OTel is not installed."""

    def set_attribute(self, key: str, value: object) -> None: ...

    def set_attributes(self, attributes: dict[str, object]) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...

    def set_status(self, *args: object, **kwargs: object) -> None: ...

    def end(self) -> None: ...


class _NoopTracer:
    """Tracer stand-in whose spans do nothing (zero-overhead when off)."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: object) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


_NOOP_TRACER = _NoopTracer()


def _resolve_otlp_target(
    settings: ObservabilitySettings,
) -> tuple[str, dict[str, str]]:
    """Endpoint + headers for the OTLP exporter.

    Convenience: when ``langfuse_*`` creds are set, derive the OTLP traces
    endpoint and a Basic-auth header from them — but explicit ``endpoint`` /
    ``headers`` always win, so a plain vendor-neutral export is unaffected.
    """
    endpoint = settings.endpoint
    headers = dict(settings.headers)
    pk = settings.langfuse_public_key
    sk = settings.langfuse_secret_key
    host = settings.langfuse_host
    if host and pk and sk:
        if not endpoint:
            endpoint = host.rstrip("/") + "/api/public/otel/v1/traces"
        if "Authorization" not in headers:
            token = base64.b64encode(f"{pk}:{sk.get_secret_value()}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
    return endpoint, headers


def init_tracing(
    settings: ObservabilitySettings,
    *,
    span_processor: SpanProcessor | None = None,
) -> bool:
    """Build and install the TracerProvider.

    Args:
        settings: Observability config. When ``enabled`` is false or
            ``exporter`` is ``"none"``, this is a no-op.
        span_processor: Injectable processor (tests pass an in-memory one).
            Defaults to a ``BatchSpanProcessor`` over an OTLP/HTTP exporter.

    Returns:
        True if a real provider was installed; False when disabled, the
        exporter is ``none``, or the ``[otel]`` extra is not installed.
    """
    global _provider
    if not settings.enabled or settings.exporter == "none":
        return False
    if not _OTEL_AVAILABLE:
        logger.warning("observability_enabled_but_otel_not_installed")
        return False

    # Idempotent: a re-init without an intervening shutdown would otherwise
    # orphan the previous provider's export thread + OTLP socket.
    if _provider is not None:
        shutdown_tracing()

    from everos import __version__

    resource = Resource.create(
        {"service.name": settings.service_name, "service.version": __version__}
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.sample_rate)),
    )
    endpoint, headers = _resolve_otlp_target(settings)
    processor = span_processor or BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint, headers=headers)
    )
    provider.add_span_processor(processor)
    _provider = provider

    from .attributes import set_capture_content

    set_capture_content(settings.capture_content)
    logger.info(
        "tracing_initialized",
        service_name=settings.service_name,
        capture_content=settings.capture_content,
    )
    return True


def get_tracer(name: str) -> Any:
    """Return a tracer for ``name`` — the real one if initialized, else no-op."""
    if _provider is None:
        return _NOOP_TRACER
    return _provider.get_tracer(name)


def force_flush(timeout_millis: int = 5000) -> None:
    """Flush pending spans. No-op (and never raises) when uninitialized."""
    if _provider is None:
        return
    try:
        _provider.force_flush(timeout_millis)
    except Exception:  # pragma: no cover - telemetry must never break callers
        logger.warning("tracing_force_flush_failed", exc_info=True)


def shutdown_tracing() -> None:
    """Flush + tear down the provider; safe to call when uninitialized."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.force_flush()
        _provider.shutdown()
    except Exception:  # pragma: no cover - telemetry must never break callers
        logger.warning("tracing_shutdown_failed", exc_info=True)
    finally:
        _provider = None
        from .attributes import set_capture_content

        set_capture_content(False)
