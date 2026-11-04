"""User-side memory types: episodes, atomic facts, foresights and profiles."""

from __future__ import annotations

__all__ = [
    "AtomicFact",
    "Episode",
    "ExplicitInfoItem",
    "Foresight",
    "ImplicitTraitItem",
    "Profile",
]

from pydantic import BaseModel, ConfigDict


class Episode(BaseModel):
    """User-side episodic memory — a structured 'what happened' trace.

    ``owner_id`` is the user this episode belongs to, or ``None`` for whole-memcell (generic) episodes
    that do not bind to any user. LLM output maps ``title`` → ``subject``, ``content`` → ``episode`` and
    ``summary`` → ``summary``. ``extra="allow"`` surfaces any LLM-emitted secondary fields without a
    schema bump.

    ``summary`` is a display preview of ``episode``: faithful to it, readable without it, introducing
    nothing it does not already say. Required rather than defaulted so that a construction site which
    forgets it fails the type-checkers instead of silently producing an empty preview — it was a blind
    ``episode[:200]`` slice for the whole of 0.1 through 0.4, which is what a silent default degrades back into.
    """

    owner_id: str | None
    episode: str
    subject: str = ""
    summary: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="allow")


class Foresight(BaseModel):
    """User-side foresight memory — anticipated future event or commitment.

    LLM output maps ``content`` → ``foresight``. Date fields are ISO ``YYYY-MM-DD`` strings.
    ``extra="allow"`` surfaces any LLM-emitted secondary fields without a schema bump.
    """

    owner_id: str
    foresight: str
    evidence: str
    timestamp: int  # Unix epoch milliseconds
    start_time: str | None = None  # YYYY-MM-DD
    end_time: str | None = None  # YYYY-MM-DD
    duration_days: int | None = None

    model_config = ConfigDict(extra="allow")


class AtomicFact(BaseModel):
    """User-side atomic fact — a single verifiable assertion lifted from a conversation.

    ``owner_id`` is the user this fact belongs to, or ``None`` for whole-memcell (generic) facts that do not bind to any user.
    Secondary LLM fields (confidence, sources, topic, …) land via ``extra="allow"`` without a schema bump.
    """

    owner_id: str | None
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="allow")


class ExplicitInfoItem(BaseModel):
    """One ``Profile.explicit_info`` item: a single fact a source states about the owner.

    ``extra="allow"`` lets a host attach its own keys (ids, provenance); they ride through every merge untouched.

    Attributes:
        description: The fact as one short sentence. Its case- and whitespace-folded form is the item's identity.
        category: The dimension the fact is filed under; many items share one.
        evidence: Excerpt or faithful paraphrase of the source that grounds the fact.
        observed_at: Unix epoch milliseconds of the source that last established ``description``. The extractor
            writes it; a host persists it and passes it back unchanged.
    """

    description: str
    category: str | None = None
    evidence: str | None = None
    observed_at: int | None = None

    model_config = ConfigDict(extra="allow")


class ImplicitTraitItem(BaseModel):
    """One ``Profile.implicit_traits`` item: a stable pattern inferred across sources.

    ``extra="allow"`` lets a host attach its own keys (ids, provenance); they ride through every merge untouched.

    Attributes:
        description: The pattern as one short sentence. Its case- and whitespace-folded form is the item's identity.
        trait: The trait label the pattern is filed under; many items may share one.
        basis: The signals the inference rests on.
        observed_at: Unix epoch milliseconds of the source that last established ``description``. The extractor
            writes it; a host persists it and passes it back unchanged.
    """

    description: str
    trait: str | None = None
    basis: str | None = None
    observed_at: int | None = None

    model_config = ConfigDict(extra="allow")


class Profile(BaseModel):
    """User-level aggregate of long-term facts and traits — not anchored to a single MemCell.

    ``extra="allow"`` keeps any other top-level key a host attaches.
    """

    owner_id: str
    summary: str
    timestamp: int  # Unix epoch milliseconds
    explicit_info: list[ExplicitInfoItem] = []
    implicit_traits: list[ImplicitTraitItem] = []

    model_config = ConfigDict(extra="allow")
