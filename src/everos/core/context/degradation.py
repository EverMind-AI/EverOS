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

WHAT THE CONTEXTVAR HOLDS, AND WHY IT IS A LIST. The variable holds a mutable sink, and
:func:`mark_degraded` appends to it; it never rebinds the variable. Rebinding is the
obvious implementation and it does not work here: ``asyncio.gather`` runs each argument
in a Task, a Task starts from a *copy* of the current context, and a ``set()`` inside
that copy is invisible to the parent that reads the result. ``SearchManager.search``
gathers the episode route -- the one that owns the multi-round decider -- so with a
rebinding implementation every degraded search still returned ``degraded == []``, which
is precisely the invisibility this module exists to remove. A copied context copies the
*binding*, not the object it points at, so a shared sink crosses the task boundary while
a fresh sink per :func:`reset_degradations` keeps one request out of the next one.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_degradations: ContextVar[list[str] | None] = ContextVar(
    "everos_degradations", default=None
)


def mark_degraded(reason: str) -> None:
    """Record that this request's result came from a degraded path.

    Repeats are collapsed: a per-question loop hits the same fallback on every round,
    and the response should say *what* degraded, not how many times.

    A no-op when no sink is installed -- outside a request nothing will ever read the
    reason, and lazily installing one here would put it in whichever task happened to
    call first, which is the bug this module is written to avoid.
    """
    reason = (reason or "").strip()
    if not reason:
        return
    sink = _degradations.get()
    if sink is None:
        return
    if reason not in sink:
        sink.append(reason)


def get_degradations() -> tuple[str, ...]:
    """Reasons recorded for the current request, in the order first seen."""
    sink = _degradations.get()
    return tuple(sink) if sink else ()


def reset_degradations() -> Token[list[str] | None]:
    """Install a fresh sink and return a token for restoring the previous one.

    Called at the start of a request. Without it a long-lived task's context would
    accumulate reasons across requests and mark healthy results degraded.
    """
    return _degradations.set([])


def restore_degradations(token: Token[list[str] | None]) -> None:
    """Restore whatever was recorded before the matching reset."""
    _degradations.reset(token)
