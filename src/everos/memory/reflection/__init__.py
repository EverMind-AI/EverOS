"""Reflection — offline memory consolidation.

Merges fragmented cluster members into higher-quality episodes and
deprecates the originals. Two steps: the orchestrator merges and writes; the
finisher deprecates on the merged episode's event.

External usage:
    from everos.memory.reflection import ReflectionFinisher, ReflectionOrchestrator
"""

from __future__ import annotations

from .orchestrator import ReflectionFinisher as ReflectionFinisher
from .orchestrator import ReflectionOrchestrator as ReflectionOrchestrator

__all__ = ["ReflectionFinisher", "ReflectionOrchestrator"]
