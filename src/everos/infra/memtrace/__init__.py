"""MemTrace file-tracking instrumentation (optional smartcomment tracing).

Public facade for the MemTrace execution-graph tracer. Application bootstrap
injects :class:`MemTraceTracker` into the memory service for the
``/memory/add``, ``/memory/flush`` and ``/memory/search`` boundaries; see
:mod:`everos.infra.memtrace.tracker` for the tracing strategy and configuration.
"""

from __future__ import annotations

from .tracker import MemTraceTracker as MemTraceTracker

__all__ = ["MemTraceTracker"]
