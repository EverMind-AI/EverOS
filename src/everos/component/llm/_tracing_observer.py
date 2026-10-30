"""Observer that puts completion telemetry on the active OpenTelemetry span.

Token usage accumulates onto the current span via ``set_generation_usage``
(a no-op without an active recording span). A finish reason other than
``stop`` is logged as a warning whether or not tracing records anything:
truncations and aborts are what an operator needs to triage first.
"""

from __future__ import annotations

__all__ = ["TracingObserver"]

from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import set_generation_usage

from .observer import TokenUsage

logger = get_logger(__name__)


class TracingObserver:
    """:class:`LLMCallObserver` that records usage on the span and flags odd stops."""

    def on_completion(
        self,
        *,
        model: str,
        usage: TokenUsage | None,
        finish_reason: str | None,
        attempt: int,
    ) -> None:
        """Record ``usage`` on the active span; warn on a non-``stop`` finish."""
        set_generation_usage(
            model=model,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )
        if finish_reason is not None and finish_reason != "stop":
            logger.warning(
                "llm_non_stop_finish",
                model=model,
                finish_reason=finish_reason,
                attempt=attempt,
            )
