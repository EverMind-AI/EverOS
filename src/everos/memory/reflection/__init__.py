"""Reflection — offline memory consolidation.

Merges fragmented cluster members into higher-quality episodes, re-
extracts atomic facts, and deprecates the originals.

External usage:
    from everos.memory.reflection import ReflectionOrchestrator
"""

from __future__ import annotations

from .orchestrator import ReflectionOrchestrator as ReflectionOrchestrator

__all__ = ["ReflectionOrchestrator"]
