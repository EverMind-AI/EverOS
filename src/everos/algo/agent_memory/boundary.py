"""Boundary detection for agent memory: :class:`everos.algo.boundary.BoundaryDetector`, re-exported unchanged.

Agent trajectories mix ``ChatMessage`` with ``ToolCallRequest`` / ``ToolCallResult``; ``BoundaryDetector.adetect``
takes them as they are and keeps the tool items in the cells it returns.
"""

from __future__ import annotations

__all__ = ["BoundaryDetector"]

from everos.algo.boundary import BoundaryDetector
