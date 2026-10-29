"""Observability namespace: ``logging``, ``metrics`` and ``tracing`` subpackages.

Import from the subpackage you need, e.g.
``from everos.core.observability.logging import get_logger``. Nothing is
re-exported here because ``tracing`` attempts to import the optional
OpenTelemetry SDK (the ``[otel]`` extra) at import time; a re-export would put
that attempt on every observability import path.
"""

from __future__ import annotations

__all__: list[str] = []
