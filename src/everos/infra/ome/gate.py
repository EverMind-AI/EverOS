"""A check the engine runs before every strategy attempt.

The engine does not know what a purge is; the service layer hands it a
:class:`RunGate` (``OfflineEngine.set_run_gate``) that, per event, says to
run it, to drop it (it is stale), or to wait until the gate opens. Waiting
gives up the attempt's execution slot and does not count toward the run
timeout.
"""

from __future__ import annotations

__all__ = ["RunGate", "Verdict"]

from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Literal, Protocol

type Verdict = Literal["run", "discard", "wait"]


class RunGate(Protocol):
    """Decides, per event, whether a strategy attempt may run now."""

    def verdict(self, event: object) -> Verdict:
        """``run`` it, ``discard`` it (recorded as done), or ``wait``."""
        ...

    def lifted(self, event: object, run_id: str) -> AbstractAsyncContextManager[None]:
        """Entered once ``event`` may be looked at again (``run_id`` waits)."""
        ...

    def bind(self, event: object) -> AbstractContextManager[None]:
        """Held around the strategy body of an event allowed to run."""
        ...
