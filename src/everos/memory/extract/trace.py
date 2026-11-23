"""Request-local construction evidence; no tracing or persistence dependency."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from everos.core.utils.datetime import get_utc_now

_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "everos_construction_events", default=None
)


def record_construction(
    phase: str,
    *,
    snapshot_factory: Callable[[], dict[str, Any]] | None = None,
    **values: Any,
) -> None:
    """Snapshot a real boundary or extractor result, including empty results."""
    events = _EVENTS.get()
    if events is None:
        return
    try:
        if snapshot_factory is not None:
            values.update(snapshot_factory())
        events.append(
            {
                "phase": phase,
                "observed_at": get_utc_now().isoformat(),
                **copy.deepcopy(values),
            }
        )
    except Exception:
        return


@contextmanager
def collect_construction_trace() -> Iterator[list[dict[str, Any]]]:
    """Isolate each request while allowing its pipeline tasks to share evidence."""
    events: list[dict[str, Any]] = []
    token = _EVENTS.set(events)
    try:
        yield events
    finally:
        _EVENTS.reset(token)
