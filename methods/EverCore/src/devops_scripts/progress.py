"""
Devops script progress reporting

Provides a standard interface for devops scripts to emit structured progress
lines that can be parsed by parent processes (e.g., MilvusRebuildService).

Protocol:
    Scripts write lines to stdout in the format:
        ##DEVOPS## {"event": "...", "key": "value", ...}
    The parent process reads stdout line by line and parses these lines.

Usage:
    from devops_scripts.progress import StdoutProgressReporter

    progress = StdoutProgressReporter()
    progress.emit({"event": "start", "total": 6})
    progress.emit({"event": "collection_done", "alias": "v1_ep", "status": "ok"})
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict


# Standard prefix — parsed by the parent process (e.g., MilvusRebuildService)
PROGRESS_PREFIX = "##DEVOPS##"


class ProgressReporter(ABC):
    """Abstract interface for reporting devops task progress."""

    @abstractmethod
    def emit(self, data: Dict[str, Any]) -> None:
        """
        Emit a progress event.

        Args:
            data: Event payload dict. Must contain an "event" key.
        """
        ...


class StdoutProgressReporter(ProgressReporter):
    """
    Default implementation: writes structured JSON lines to stdout.

    Format: ##DEVOPS## {"event": "...", ...}
    """

    def emit(self, data: Dict[str, Any]) -> None:
        print(f"{PROGRESS_PREFIX} {json.dumps(data, ensure_ascii=False)}", flush=True)


class NoopProgressReporter(ProgressReporter):
    """Silent reporter — does nothing. Useful for tests or CLI-only runs."""

    def emit(self, data: Dict[str, Any]) -> None:
        pass
