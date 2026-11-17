"""The one exception :class:`OpenAIProvider` raises when a chat call fails.

The everalgo ``LLMClient`` contract leaves the implementation's exception type
to the host (``llm-contract.md`` §2.5): operators let it propagate unchanged,
so the host recognises its own type. ``reason`` tells the host *why* the call
failed, which is what decides whether retrying the operator is worth it.
"""

from __future__ import annotations

__all__ = ["FailedAttempt", "LLMCallError", "LLMCallFailureReason"]

from collections.abc import Sequence
from typing import Literal, NamedTuple

from everos.core.errors import LLMServiceError

LLMCallFailureReason = Literal[
    "refusal", "cut_short", "no_content", "invalid_reply", "transport"
]


class FailedAttempt(NamedTuple):
    """One attempt of a call that failed where resending it might have helped.

    Attributes:
        reason: ``invalid_reply`` (the reply failed validation),
            ``cut_short`` (the reply was cut at ``max_tokens`` on a call that
            retries a cut) or ``transport`` (no choice, or a generation
            aborted upstream).
        reply: The reply's text, ``None`` when the upstream sent none.
    """

    reason: LLMCallFailureReason
    reply: str | None


class LLMCallError(LLMServiceError):
    """A chat call failed; ``reason`` says how.

    Subclasses :class:`~everos.core.errors.LLMServiceError` so the API layer's
    existing error-to-status mapping covers it unchanged.

    Reasons:
        ``refusal``: the upstream flagged a refusal (contract §3.4).
        ``cut_short``: the upstream reported ``length`` / ``content_filter``;
            the identical request hits the same cap again (contract §3.5).
        ``no_content``: the upstream answered without content.
        ``invalid_reply``: no reply validated against ``response_format``
            within the attempt budget (contract §7.1).
        ``transport``: the SDK request failed, or the upstream kept answering
            with no choice or an aborted generation.

    Args:
        message: Human-readable detail; never carries request content.
        reason: Failure category the host branches on.
        failed_attempts: When the call failed because its attempts ran out,
            each attempt, in order, with its reply: what a host that salvages
            a reply reads. Empty when a failure ended the call at once. Never
            part of the message.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: LLMCallFailureReason,
        failed_attempts: Sequence[FailedAttempt] = (),
    ) -> None:
        super().__init__(message)
        self.reason: LLMCallFailureReason = reason
        self.failed_attempts: tuple[FailedAttempt, ...] = tuple(failed_attempts)
