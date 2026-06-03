"""Main extraction pipelines — one per track.

External usage:
    from everos.memory.extract.pipeline import (
        Pipeline, UserMemoryPipeline, AgentMemoryPipeline,
    )

Calls everalgo (the algorithm library) for boundary detection and
synchronous Episode extraction. Agent track is currently stubbed —
returns ``status="skipped"`` until the algo extractors land.
"""

from .agent_memory import AgentMemoryPipeline as AgentMemoryPipeline
from .base import Pipeline as Pipeline
from .user_memory import UserMemoryPipeline as UserMemoryPipeline

__all__ = [
    "AgentMemoryPipeline",
    "Pipeline",
    "UserMemoryPipeline",
]
