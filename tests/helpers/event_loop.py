"""The event loop for tests that drive one loop through many runs.

A Hypothesis state machine calls ``loop.run_until_complete`` once per rule,
hundreds of times per example, on a loop it owns. On Windows the default loop
is a ``ProactorEventLoop``, and each run arms its self-pipe read at the start
and cancels it at the end (``windows_events.ProactorEventLoop.run_forever``).
Now and then that read completes with WinError 995 (operation aborted);
``_loop_self_reading`` logs "Error on reading from the event loop self pipe"
and does not arm it again, so every later ``call_soon_threadsafe`` wakeup is
lost. aiosqlite and LanceDB hand results back that way, and a run waiting on
one with no timer pending sleeps for good: the Windows CI hangs in
``test_worker_stateful`` (2026-11-05) and ``test_memorize_queue_stateful``
(2026-11-24). A probe of that pattern on Windows CI (one loop, many runs, one
aiosqlite query each) hung in 8 of 60 jobs of 50,000 runs; the 7 hangs caught
with a dump all showed the read aborted by WinError 995, a callback queued and
no timer.

``asyncio.SelectorEventLoop`` wakes through ``select()`` on a socketpair and
has no read to cancel between runs; the same probe on it ran 20 jobs of 50,000
runs without a hang. The product never restarts a loop (the
server runs one loop throughout; the CLI runs ``asyncio.run`` once), so only
tests that run a loop many times need this.
"""

from __future__ import annotations

import asyncio
import sys

__all__ = ["new_stepped_loop"]


def new_stepped_loop() -> asyncio.AbstractEventLoop:
    """A new event loop that stays awake across many ``run_until_complete`` calls."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
