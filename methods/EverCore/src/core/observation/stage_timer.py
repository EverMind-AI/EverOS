from __future__ import annotations

import functools
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SLOW_THRESHOLD_MS = int(os.environ.get("STAGE_TIMER_SLOW_THRESHOLD_MS", "0"))


class _StageNode:
    """A single node in the stage timing tree.

    Each node records its name, duration, whether it represents parallel
    execution, and any child nodes.
    """

    __slots__ = ("name", "duration_ms", "is_parallel", "children")

    def __init__(self, name: str, is_parallel: bool = False) -> None:
        self.name: str = name
        self.duration_ms: float = 0.0
        self.is_parallel: bool = is_parallel
        self.children: list[_StageNode] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert this node to a JSON-serializable dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "duration_ms": round(self.duration_ms),
        }
        if self.is_parallel:
            result["parallel"] = True
        if self.children:
            result["stages"] = [child.to_dict() for child in self.children]
        return result


_current_node: ContextVar[Optional[_StageNode]] = ContextVar(
    "_stage_timer_current_node", default=None
)


class StageTimer:
    """Hierarchical timer that records stage durations for a single request.

    Usage::

        timer = StageTimer("/api/v1/memories/search")
        with timer.stage("retrieval"):
            with timer.parallel("sources"):
                ...  # parallel branches
        timer.log_summary()
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint: str = endpoint
        self._root: _StageNode = _StageNode("_root")
        self._start: float = perf_counter()

    @contextmanager
    def stage(self, name: str):
        """Time a sequential stage. Nests under the current parent."""
        node = _StageNode(name)
        parent = _current_node.get()
        if parent is None:
            parent = self._root
        parent.children.append(node)
        token = _current_node.set(node)
        start = perf_counter()
        try:
            yield
        finally:
            node.duration_ms = (perf_counter() - start) * 1000
            _current_node.reset(token)

    @contextmanager
    def parallel(self, name: str):
        """Time a parallel stage. Children are rendered with ``|`` separator."""
        node = _StageNode(name, is_parallel=True)
        parent = _current_node.get()
        if parent is None:
            parent = self._root
        parent.children.append(node)
        token = _current_node.set(node)
        start = perf_counter()
        try:
            yield
        finally:
            node.duration_ms = (perf_counter() - start) * 1000
            _current_node.reset(token)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary dict."""
        total_ms = round((perf_counter() - self._start) * 1000)
        result: dict[str, Any] = {"endpoint": self._endpoint, "total_ms": total_ms}
        if self._root.children:
            result["stages"] = [child.to_dict() for child in self._root.children]
        return result

    def log_summary(self) -> None:
        """Log the timer summary as JSON at INFO level."""
        s = self.summary()
        if 0 < _SLOW_THRESHOLD_MS <= s["total_ms"]:
            s["slow"] = True
        logger.info(
            "[stage_timer] %s", json.dumps(s, ensure_ascii=False, separators=(",", ":"))
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def start_timer(endpoint: str) -> None:
    """Create a StageTimer and store it in the current request's app_info context."""
    from core.context.context import get_current_app_info

    app_info = get_current_app_info()
    if app_info is not None:
        app_info["stage_timer"] = StageTimer(endpoint)


def log_timer() -> None:
    """Log the current request's StageTimer summary, if one exists."""
    timer = get_current_timer()
    if timer is not None:
        timer.log_summary()


def stage_timed(endpoint: str):
    """Decorator that wraps an async controller method with StageTimer.

    Calls start_timer() before the method and log_timer() after it returns.
    On exception, log_timer() is NOT called (consistent with GlobalExceptionHandler).
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_timer(endpoint)
            result = await func(*args, **kwargs)
            log_timer()
            return result

        return wrapper

    return decorator


def get_current_timer() -> Optional[StageTimer]:
    """Return the StageTimer attached to the current request, or None."""
    from core.context.context import get_current_app_info

    app_info = get_current_app_info()
    if app_info is None:
        return None
    return app_info.get("stage_timer")


@contextmanager
def timed(name: str):
    """Convenience context manager that records a stage on the current timer.

    No-op when no timer is present in the request context.
    """
    timer = get_current_timer()
    if timer is None:
        yield
        return
    with timer.stage(name):
        yield


@contextmanager
def timed_parallel(name: str):
    """Convenience context manager that records a parallel stage on the current timer.

    No-op when no timer is present in the request context.
    """
    timer = get_current_timer()
    if timer is None:
        yield
        return
    with timer.parallel(name):
        yield
