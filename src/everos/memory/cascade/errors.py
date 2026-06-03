"""Cascade error hierarchy — drives the worker's retry classification.

The worker decides ``mark_failed(retryable=True/False)`` purely by
exception class:

- :class:`RecoverableError` → transient (HTTP 5xx, network, embedding
  rate limit). Worker retries up to ``MAX_RETRY`` inline, then marks
  ``retryable=TRUE`` so ``cascade fix --apply`` can re-enqueue.
- :class:`UnrecoverableError` → fatal (YAML parse, missing required
  field, schema mismatch). Worker stops immediately and marks
  ``retryable=FALSE`` — only a user edit to the md will unstick it.
- Anything else → treated as :class:`UnrecoverableError`. The worker
  catches ``Exception`` defensively so an unexpected failure never
  hangs the daemon, but the diagnostic message carries the original
  type for triage.
"""

from __future__ import annotations


class CascadeError(Exception):
    """Root of the cascade error tree."""


class RecoverableError(CascadeError):
    """Transient failure — worker should retry then mark retryable."""


class UnrecoverableError(CascadeError):
    """Fatal failure — needs a user edit before re-running."""
