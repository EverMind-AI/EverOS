"""Stamp every write request with its arrival time and track it while it runs.

A purge drops background work whose input predates it, and waits for the
work already in flight before it clears the scope a second time. A write
request is such an input: its arrival time becomes the request's ``ORIGIN``
(events it emits carry it), and the request is registered with the purge
for as long as it runs. Nothing is refused here (a write into the range
being purged is refused by its route); ``GET`` / ``HEAD`` / ``OPTIONS`` pass
untouched.
"""

from __future__ import annotations

__all__ = ["WriteTrackingMiddleware"]

from starlette.types import ASGIApp, Receive, Scope, Send

from everos.core.purge_marks import ORIGIN
from everos.core.utils.datetime import get_utc_now

_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class WriteTrackingMiddleware:
    """Sets ``ORIGIN`` for a write request and tracks it with the purge."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        application = getattr(scope["app"].state, "application", None)
        if (
            scope["type"] != "http"
            or scope["method"] in _READ_METHODS
            or application is None
        ):
            await self.app(scope, receive, send)
            return
        origin = get_utc_now()
        token = ORIGIN.set(origin)
        try:
            with application.purge.track(origin):
                await self.app(scope, receive, send)
        finally:
            ORIGIN.reset(token)
