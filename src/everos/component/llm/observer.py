"""Hook for observing every completion :class:`OpenAIProvider` receives.

The contract's ``achat`` returns only the reply text, so token usage and the
finish reason are visible inside the provider and nowhere else. Rather than
teach the provider about OpenTelemetry, it reports each completion to an
injected :class:`LLMCallObserver`; failed attempts are reported too, because
their tokens were billed all the same.
"""

from __future__ import annotations

__all__ = ["LLMCallObserver", "NullObserver", "TokenUsage"]

from typing import NamedTuple, Protocol


class TokenUsage(NamedTuple):
    """Token counts the upstream reported for one completion.

    Attributes:
        input_tokens: Prompt tokens, or ``None`` if not reported.
        output_tokens: Completion tokens, or ``None`` if not reported.
    """

    input_tokens: int | None
    output_tokens: int | None


class LLMCallObserver(Protocol):
    """Receives one notification per completion, whatever its outcome."""

    def on_completion(
        self,
        *,
        model: str,
        usage: TokenUsage | None,
        finish_reason: str | None,
        attempt: int,
    ) -> None:
        """Record one completion.

        Args:
            model: Model id the upstream reports having served.
            usage: Reported token counts; ``None`` if the upstream sent none.
            finish_reason: Reported finish reason, verbatim; ``None`` if absent.
            attempt: 1-based attempt number within one ``achat`` call.
        """
        ...


class NullObserver:
    """Observer that discards every notification."""

    def on_completion(
        self,
        *,
        model: str,
        usage: TokenUsage | None,
        finish_reason: str | None,
        attempt: int,
    ) -> None:
        """Discard the notification."""
