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
    "EpisodeIndex",
    "GroupProfileIndex",
    "ImageMemoryIndex",
    "IndexEraser",
    "KnowledgeTopicIndex",
    "ProfileIndex",
]

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, runtime_checkable

from everos.memory.models import (
    AgentCase,
    AgentSkill,
    Episode,
    EpisodeParentType,
    GroupProfileItem,
    ImageMemory,
    Profile,
)
from everos.memory.ports.query import (
    Filter,
    MemoryRef,
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
class ImageMemoryIndex(Protocol):
    """Read side of the picture memories — a peer of :class:`EpisodeIndex`.

    A picture has two embeddings of different provenance, so unlike an episode
    it gets three ANN-comparable routes: BM25 across the description and the
    text in the picture, ANN over the description's text embedding
    (``vector``), and ANN over the picture bytes' own embedding
    (``visual_vector``). The last two live in different vector spaces — a
    cosine between them means nothing — so they are recalled separately and
    fused by rank, never compared directly.
    """

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[ImageMemory]]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[ImageMemory]]: ...

    async def visual_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[ImageMemory]]:
        """ANN over ``visual_vector`` (the picture's own embedding).

        Empty when no vision embedder is configured: every row's
        ``visual_vector`` is NULL and the column has nothing to score.
        """
        ...

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[ImageMemory]]: ...

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[ImageMemory]]:
        """The named entries, for re-reading what a forget has to rewrite."""
        ...

    async def find_by_parent(
        self, *, parent_type: str, parent_id: str, limit: int = 100
    ) -> list[Stored[ImageMemory]]:
        """Pictures that came in with one source segment.

        This is the edge a forget walks: forgetting a segment retires the
        pictures that arrived with it.
        """
        ...

    async def find_by_image_ids(
        self, scope: Scope, image_ids: Sequence[str]
    ) -> list[Stored[ImageMemory]]:
        """The *current* picture memories about these files, one per occurrence.

        What an episode's ``image_ids`` resolve through. Only current rows come
        back: a forgotten or superseded picture must not be resurrected into an
        answer just because an episode still refers to it — the reference is
        history, the content is not current.
        """
        ...


@runtime_checkable
class GroupProfileIndex(Protocol):
    """Read side of the group-profile items (table ``group_profile_item``).

    Rows carry the owner and window columns every group record has
    (``owner_type`` / ``owner_id`` / ``app_id`` / ``project_id`` /
    ``deprecated_by`` / ``source_start`` / ``source_end``), so a read scope's
    predicate applies to them unchanged, plus the link columns a read follows:
    ``entry_id``, ``kind``, ``topic_id``, ``supersedes``, ``about``,
    ``member_id``, ``source_memcell_ids`` (list), ``subject_ids`` (list) and
    ``observed_at``. No vector: items are found by those links and by BM25
    over their statements.

    Every item comes back with ``item_id``, and with ``status`` /
    ``superseded_by`` read from the row's ``deprecated_by`` (``forgotten``,
    ``superseded`` by that entry, or the item's own status).
    """

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[Stored[GroupProfileItem]]:
        """Items matching ``where``, newest ``observed_at`` first."""
        ...

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[GroupProfileItem]]:
        """BM25 over the items' statements, names and summaries."""
        ...

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[GroupProfileItem]]: ...


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

    async def member(self, scope: Scope, member_id: str) -> Stored[Profile] | None:
        """A member's profile inside the group ``scope`` names; ``None`` if none.

        The item is the profile about ``member_id`` (its ``owner_id``); the
        stored scope is the group's. Its items keep their provenance keys, so
        a read can drop those outside the asker's windows.
        """
        ...


@runtime_checkable
class KnowledgeTopicIndex(Protocol):
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[ScoredKnowledgeTopic]: ...

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[ScoredKnowledgeTopic]: ...


@runtime_checkable
class IndexEraser(Protocol):
    """The one write a deletion makes to the derived index: removing rows.

    The domain otherwise only reads the indexes (the cascade writes them).
    A deletion removes the rows of what it removed from md at once, instead
    of waiting for the cascade to notice, and keeps the cascade from writing
    them back. Failures are logged and handed back to the cascade, never
    raised: the markdown is already gone.
    """

    async def erase(
        self, scope: Scope, removed: Mapping[str, Sequence[MemoryRef]]
    ) -> None:
        """Remove the rows of ``removed`` entries, by daily-log table.

        Keys are table names (``episode``, ``image_memory``, ``agent_case``).
        """
        ...

    async def erase_owner(self, scope: Scope, kinds: Collection[str]) -> None:
        """Remove every row of the owner in the per-owner tables ``kinds``
        (``user_profile``, ``agent_skill``)."""
        ...

    async def requeue(self, kind: str, md_paths: Collection[str]) -> None:
        """Hand files back to the cascade so they are projected again.

        An undo re-writes a file whose bytes the index has already reconciled
        — a mark it cleared, a row erased — so the cascade's own diff sees
        nothing to do. Queuing the file forces the projection to read the file
        as it is now, which is the only way the rows come back.
        """
        ...
