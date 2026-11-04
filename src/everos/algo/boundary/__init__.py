"""Boundary detection — cut a conversation into MemCells.

Public surface:
- BoundaryDetector      — class binding an LLM client: ``adetect`` (batch) and ``adetect_step``
                          (incremental, one message at a time)
- adetect_boundaries    — async function: the batch path over ``ConversationItem`` (tool items are kept in the
                          cells but never shown to the model)
- DetectionResult       — ``(cells, tail, tail_complete)`` NamedTuple returned by all of the above
- adetect_boundary_step — async function: the single LLM decision under ``adetect_step``
- BoundaryDecision      — NamedTuple returned by ``adetect_boundary_step``

``everos.algo.user_memory`` and ``everos.algo.agent_memory`` re-export ``BoundaryDetector`` unchanged.
"""

from __future__ import annotations

import logging

from everos.algo.boundary._detector import (
    BoundaryDecision,
    BoundaryDetector,
    DetectionResult,
    adetect_boundaries,
    adetect_boundary_step,
)

__all__ = [
    "BoundaryDecision",
    "BoundaryDetector",
    "DetectionResult",
    "adetect_boundaries",
    "adetect_boundary_step",
]

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())
