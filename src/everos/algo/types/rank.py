"""Rank I/O contracts: Candidate, RankInput, ScoredItem and RankOutput.

A ranker performs no storage I/O: the caller's recall layer fetches the candidates and passes them in a
``RankInput``, and the ranker reads nothing else.
"""

from __future__ import annotations

__all__ = ["Candidate", "FactCandidate", "RankInput", "RankOutput", "ScoredItem"]

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """A single recall candidate produced by the upstream recall layer and consumed by the ranker.

    Business fields (``quality_score`` / ``maturity_score`` / ``confidence`` /
    raw text / ``parent_episode_id`` / ...) live in ``metadata``. Each ranker
    facade reads only the keys it cares about.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    source: Literal["keyword", "vector", "other"] = "other"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankInput(BaseModel):
    """Recall-stage input passed to a ranker facade.

    The multi-route recall candidates (sparse and dense; profile reads dense only) and the business parameters
    (``top_k``, ``radius``). The ranker reads no environment variable: the caller passes everything explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    memory_type: Literal["episodic", "profile", "case", "skill"]

    sparse_candidates: list[Candidate] = []
    dense_candidates: list[Candidate] = []

    top_k: int = 10
    radius: float | None = None


class ScoredItem(BaseModel):
    """A single ranked output item.

    Episodic may produce a mix of episode and atomic_fact items (expansion
    decides); other facades produce a single ``item_type``. The caller reshapes
    the list back into the response DTO it serves to clients.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    item_type: Literal["episode", "atomic_fact", "case", "skill", "profile"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_episode_id: str | None = None


class RankOutput(BaseModel):
    """Ranker output — a sorted list of ScoredItem plus optional debug metadata."""

    model_config = ConfigDict(extra="forbid")

    items: list[ScoredItem] = []
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactCandidate(BaseModel):
    """Atomic-fact candidate for episode -> fact expansion.

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
