"""The repositories of every derived-index business table of one engine."""

from __future__ import annotations

__all__ = ["BUSINESS_SCHEMAS", "IndexRepositories"]

from dataclasses import dataclass, fields
from typing import Any

from ..records import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    KnowledgeTopic,
    UserProfile,
)
from .protocols import (
    AgentSkillIndexRepository,
    EpisodeIndexRepository,
    IndexRepository,
)

BUSINESS_SCHEMAS: tuple[type[Any], ...] = (
    Episode,
    AtomicFact,
    Foresight,
    AgentCase,
    AgentSkill,
    UserProfile,
    KnowledgeTopic,
)
"""The record class of every business table, in :class:`IndexRepositories`
field order; the engine-neutral list every per-table sweep starts from."""


@dataclass(frozen=True)
class IndexRepositories:
    """One repository per business table, all on the same engine.

    Built by the engine's backend over its connection; the cascade
    projections, the domain's index adapters and the operational tooling
    take their repository from here.
    """

    episode: EpisodeIndexRepository[Any]
    atomic_fact: IndexRepository[Any]
    foresight: IndexRepository[Any]
    agent_case: IndexRepository[Any]
    agent_skill: AgentSkillIndexRepository[Any]
    user_profile: IndexRepository[Any]
    knowledge_topic: IndexRepository[Any]

    def all(self) -> tuple[IndexRepository[Any], ...]:
        """Every repository, in declaration order."""
        return tuple(getattr(self, field.name) for field in fields(self))

    def for_schema(self, schema: type[Any]) -> IndexRepository[Any]:
        """The repository whose table is declared by ``schema``.

        Raises:
            KeyError: No business table is declared by ``schema``.
        """
        for repo in self.all():
            if repo.schema is schema:
                return repo
        raise KeyError(f"no derived-index repository for {schema!r}")

    def with_vectors(self) -> dict[str, IndexRepository[Any]]:
        """Every repository whose table carries a vector column, by table name.

        The tables a row can sit in without its embedding (written while no
        embedding provider was configured) -- what backfill re-embeds.
        """
        return {
            schema.TABLE_NAME: self.for_schema(schema)
            for schema in BUSINESS_SCHEMAS
            if schema.vector_columns()
        }
