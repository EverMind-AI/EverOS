"""Search-local candidate types that everalgo no longer ships.

everalgo-core 0.6.0 removed ``everalgo.types.FactCandidate`` together with
``RankInput.episode_to_facts``. The search layer still uses it to group
prefetched atomic facts under their parent episode for the MaxSim step of
:func:`everos.memory.search.hierarchy.heap_expand`, so it lives here for now.
Whether it stays, moves into a mapper, or goes away is decided with the rest
of the search rework (alignment stage 5).
"""

from __future__ import annotations

__all__ = ["FactCandidate"]

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FactCandidate(BaseModel):
    """Atomic-fact candidate for episode -> fact expansion.

    Fields copied verbatim from ``everalgo.types.rank.FactCandidate`` as of
    everalgo-core 0.3.0.

    Attributes:
        id: Atomic-fact id.
        parent_episode_id: Episode the fact was extracted from.
        score: Cosine similarity from the vector store, prefetched by recall.
        metadata: Free-form fields the ranker may read.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_episode_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
