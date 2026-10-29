"""Main extraction pipelines — one per track, run one boundary cell at a time.

External usage:
    from everos.memory.extract.pipeline import (
        AgentMemoryPipeline, CellOutcome, PipelineOutcome, PreparedCell,
        UserMemoryPipeline,
    )

Calls everalgo (the algorithm library) for synchronous Episode extraction.
A pipeline returns the events its cell produced; the service publishes them.
"""

from __future__ import annotations

from .agent_memory import AgentMemoryPipeline as AgentMemoryPipeline
from .base import CellOutcome as CellOutcome
from .base import Pipeline as Pipeline
from .base import PipelineOutcome as PipelineOutcome
from .base import PreparedCell as PreparedCell
from .user_memory import UserMemoryPipeline as UserMemoryPipeline

__all__ = [
    "AgentMemoryPipeline",
    "CellOutcome",
    "Pipeline",
    "PipelineOutcome",
    "PreparedCell",
    "UserMemoryPipeline",
]
