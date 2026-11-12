"""Boundary detection — the session buffer cut into memcells.

External usage:
    from everos.memory.extract.boundary import BoundaryDetection, BoundaryOutcome
"""

from __future__ import annotations

from .detection import BoundaryDetection as BoundaryDetection
from .detection import BoundaryOutcome as BoundaryOutcome

__all__ = ["BoundaryDetection", "BoundaryOutcome"]
