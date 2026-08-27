"""Request-scoped record of a result that was produced by a degraded path.

Some failures inside a search do not stop it. The clearest case: when every
multi-round decider attempt fails, the loop falls back to a fixed top-N core and
stops after one round rather than returning nothing -- which is the right call for
availability, and invisible from outside. A caller receives HTTP 200, a full
``episodes`` list, and no indication that the component under test never ran.

That invisibility has a measured cost. On 2026-08-25 a twelve-arm retrieval sweep
sent a model name to an endpoint that did not serve it. Every decider call returned
404, every round fell back, and all twelve arms reported plausible accuracies between
87% and 93% -- the same degraded path twelve times. Nothing in any response said so;
it took reading the traces to find out, and the numbers had already been written up.

So a degraded result says so. Collected through a ``ContextVar`` rather than threaded
through call signatures for the same reason the request id is: the flag originates deep
in the retrieval loop and is needed at the response boundary, and every layer between
has no interest in it.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_degradations: ContextVar[tuple[str, ...]] = ContextVar(
    "everos_degradations", default=()
)


def mark_degraded(reason: str) -> None:
    """Record that this request's result came from a degraded path.

    Repeats are collapsed: a per-question loop hits the same fallback on every round,
    and the response should say *what* degraded, not how many times.
    """
    reason = (reason or "").strip()
    if not reason:
        return
    current = _degradations.get()
    if reason not in current:
        _degradations.set((*current, reason))


def get_degradations() -> tuple[str, ...]:
    """Reasons recorded for the current request, in the order first seen."""
    return _degradations.get()


def reset_degradations() -> Token[tuple[str, ...]]:
    """Clear the record and return a token for restoring it.

    Called at the start of a request. Without it a long-lived task's context would
    accumulate reasons across requests and mark healthy results degraded.
    """
    return _degradations.set(())


def restore_degradations(token: Token[tuple[str, ...]]) -> None:
    """Restore whatever was recorded before the matching reset."""
    _degradations.reset(token)
