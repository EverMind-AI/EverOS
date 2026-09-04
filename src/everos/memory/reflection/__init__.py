"""Reflection — offline memory consolidation.

Episode path: merge fragmented ``kind=user_memory`` cluster members,
re-extract atomic facts, deprecate originals.

Decision path: merge fragmented ``kind=decision`` cluster members into
one Decision, emit ``DecisionExtracted(source="reflection")``, deprecate
originals. No atomic-fact re-extract.

External usage:
    from everos.memory.reflection import (
        DecisionReflectionOrchestrator,
        ReflectionOrchestrator,
    )
"""

from __future__ import annotations

from .decision_orchestrator import (
    DecisionReflectionOrchestrator as DecisionReflectionOrchestrator,
)
from .orchestrator import ReflectionOrchestrator as ReflectionOrchestrator

__all__ = ["DecisionReflectionOrchestrator", "ReflectionOrchestrator"]
