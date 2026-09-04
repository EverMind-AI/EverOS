"""An input the embedder will never accept must not be retried.

Head-of-line blocking, measured: 8 rows carrying episodes longer than the embedding
model's context came back HTTP 400. Every provider error was wrapped as
``EmbeddingServiceError`` -- the retryable branch -- so the cascade worker retried each
one inline, slept its backoff while holding the worker slot, re-enqueued it across
scanner cycles, and the 220 healthy rows queued behind them waited. The retries could
not have helped: identical bytes to an identical endpoint return an identical 400.

The split pinned here is by status, not by exception type: 4xx is not uniformly
permanent (408 and 429 are transient) and 5xx is not uniformly transient in name only.
"""

from __future__ import annotations

import openai
import pytest

from everos.component.embedding.openai_provider import _classify
from everos.core.errors import (
    EmbeddingInputError,
    EmbeddingServiceError,
    ExternalServiceError,
    InvalidInputError,
)


class _StatusError(openai.OpenAIError):
    def __init__(self, status: int | None) -> None:
        self.status_code = status
        super().__init__(f"status {status}")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 414, 422])
def test_input_rejections_are_permanent(status: int) -> None:
    """The provider is telling us about the payload; sending it again says the same."""
    err = _classify(_StatusError(status))
    assert isinstance(err, EmbeddingInputError)
    assert not isinstance(err, ExternalServiceError)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, None])
def test_service_failures_stay_retryable(status: int | None) -> None:
    """408 and 429 are 4xx and transient -- a bare `status >= 400` rule breaks both.

    ``None`` covers transport errors (connection reset, DNS), which carry no status and
    are the most retryable case of all.
    """
    err = _classify(_StatusError(status))
    assert isinstance(err, EmbeddingServiceError)
    assert isinstance(err, ExternalServiceError)


def test_permanent_branch_is_not_under_external_service_error() -> None:
    """This is the property the cascade worker actually switches on.

    ``worker.py`` catches ``ExternalServiceError`` for the retry path and falls through
    to ``except Exception`` for permanent failure. Placing the new class anywhere under
    ``ExternalServiceError`` would restore the exact bug while looking fixed.
    """
    assert issubclass(EmbeddingInputError, InvalidInputError)
    assert not issubclass(EmbeddingInputError, ExternalServiceError)


def test_status_is_named_in_the_message() -> None:
    """`cascade fix` shows this string; without the code it cannot be triaged."""
    assert "413" in str(_classify(_StatusError(413)))
