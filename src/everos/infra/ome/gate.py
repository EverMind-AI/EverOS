"""A check the engine runs before every strategy attempt.

The engine does not know what a purge is; the service layer hands it a
:class:`RunGate` (``OfflineEngine.set_run_gate``) that, per event, says to
run it or to drop it (it is stale), and that is held around the strategy
body of an event allowed to run.
"""

from __future__ import annotations

__all__ = ["RunGate", "Verdict"]

from contextlib import AbstractContextManager
from typing import Literal, Protocol

type Verdict = Literal["run", "discard"]


class RunGate(Protocol):
    """Decides, per event, whether a strategy attempt runs."""

    def verdict(self, event: object) -> Verdict:
        """``run`` it, or ``discard`` it (recorded as done)."""
        ...

    def bind(self, event: object) -> AbstractContextManager[None]:
        """Held around the strategy body of an event allowed to run."""
        ...
