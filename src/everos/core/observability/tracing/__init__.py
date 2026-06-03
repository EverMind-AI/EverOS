"""Tracing utilities — W3C-compatible request id generation.

External usage::

    from everos.core.observability.tracing import gen_request_id
"""

from __future__ import annotations

from uuid import uuid4


def gen_request_id() -> str:
    """Generate a request id matching the W3C trace-context spec.

    Returns 32 lowercase hex characters (128-bit, no prefix) — the same
    format as a W3C ``trace_id`` / OpenTelemetry trace identifier. Routes
    and services that mint a fresh request id (when one wasn't injected
    by upstream middleware) should call this helper rather than rolling
    their own uuid / prefix format, so the id layer stays compatible
    with OpenTelemetry exporters and standard APM tooling.

    Example::

        >>> rid = gen_request_id()
        >>> len(rid)
        32
    """
    return uuid4().hex


__all__ = ["gen_request_id"]
