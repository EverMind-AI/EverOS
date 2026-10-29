"""Resolution of persisted event topics back to their classes.

The event types themselves are declared by the domain in
:mod:`everos.memory.ports`; the engine only needs the inverse of
``Event.topic()`` to rehydrate a run's payload during dispatch and crash
recovery.
"""

from __future__ import annotations

__all__ = ["resolve_topic"]

import importlib
from functools import cache
from typing import Any

from everos.memory.ports import Event


@cache
def resolve_topic(topic: str) -> type[Event]:
    """Inverse of ``Event.topic()``; imports and returns the class.

    Cached because crash recovery may resolve the same topic many times in
    a tight loop, and ``importlib.import_module`` is non-trivial.

    Raises:
        ValueError: ``topic`` is not ``"<module>:<class>"``.
        TypeError: The topic does not name an :class:`Event` subclass.
    """
    module_name, sep, cls_name = topic.partition(":")
    if not sep or not cls_name:
        raise ValueError(f"invalid event topic: {topic!r}")
    mod: Any = importlib.import_module(module_name)
    cls = getattr(mod, cls_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Event)):
        raise TypeError(f"topic {topic!r} did not resolve to an Event subclass")
    return cls
