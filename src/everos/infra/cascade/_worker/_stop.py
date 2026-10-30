"""The interruptible sleep every long-lived worker loop waits on.

The worker owns one ``asyncio.Event`` that :meth:`CascadeWorker.stop` sets;
the drain loop, the loop supervisor and the index maintainer all share it, so
every wait in the subsystem ends as soon as shutdown begins instead of
outlasting its own cooldown.
"""

from __future__ import annotations

import asyncio


async def wait_or_stop(stop: asyncio.Event, seconds: float) -> bool:
    """Sleep up to ``seconds``; return True if ``stop`` was set."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True
