"""Read-side storage ports: the rebuildable derived indexes.

Indexes are projections of the markdown truth maintained by the cascade;
the domain only reads them. Every read returns domain objects: search
methods as scored :class:`SearchHit` matches (owner-less knowledge topics
as :class:`ScoredKnowledgeTopic`), other reads wrapped in :class:`Stored`.
Column names, scoring columns and vector fields stay inside the adapters.
"""

from __future__ import annotations

__all__ = [
    "AgentCaseIndex",
    "AgentSkillIndex",
    "AtomicFactIndex",
    "EpisodeIndex",
    "KnowledgeTopicIndex",
    "ProfileIndex",
]

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    EpisodeParentType,
    Profile,
)
from everos.memory.ports.query import (
    Filter,
    Page,
    PageResult,
    Scope,
    ScoredKnowledgeTopic,
    SearchHit,
    Stored,
)


@runtime_checkable
class EpisodeIndex(Protocol):
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[Episode]]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[Episode]]: ...

    async def dense_search_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[Episode]]:
        """ANN over the subject embedding; episodes without one never match."""
        ...

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[Episode]]:
        """Every match when ``limit`` is ``None``."""
        ...

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[Episode]]: ...

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[Episode]]: ...

    async def count(self, scope: Scope, *, parent_type: EpisodeParentType) -> int: ...

    async def list_after(
        self,
        scope: Scope,
        after_ts: int,
        *,
        parent_type: EpisodeParentType,
        limit: int | None = None,
    ) -> list[Stored[Episode]]:
        """Episodes newer than ``after_ts`` (epoch milliseconds)."""
        ...

    async def orphans(self, where: Filter, *, limit: int) -> list[Stored[Episode]]: ...


@runtime_checkable
class AtomicFactIndex(Protocol):
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[AtomicFact]]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[AtomicFact]]: ...

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[AtomicFact]]: ...


@runtime_checkable
class AgentCaseIndex(Protocol):
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[AgentCase]]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[AgentCase]]: ...

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[AgentCase]]: ...

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[AgentCase]]: ...


@runtime_checkable
class AgentSkillIndex(Protocol):
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[AgentSkill]]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[AgentSkill]]: ...

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[AgentSkill]]: ...

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[AgentSkill]]: ...

    async def count(self) -> int: ...

    async def count_in_cluster(self, scope: Scope, cluster_id: str) -> int: ...

    async def top_k_in_cluster(
        self, scope: Scope, cluster_id: str, vector: Sequence[float], k: int
    ) -> list[Stored[AgentSkill]]:
        """The ``k`` skills of the cluster closest to ``vector``."""
        ...


@runtime_checkable
class ProfileIndex(Protocol):
    async def get(self, scope: Scope) -> Stored[Profile] | None: ...


@runtime_checkable
class KnowledgeTopicIndex(Protocol):
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[ScoredKnowledgeTopic]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[ScoredKnowledgeTopic]: ...
