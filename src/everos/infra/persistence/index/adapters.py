"""Backend-neutral adapters from the derived-index repositories to the ports.

Both engines expose the same :class:`IndexRepository` contract, so one
adapter per memory kind serves either;
:func:`everos.infra.persistence.factories.build_indexes` binds each to the
selected engine's repository. The adapter
owns the row -> domain conversion and the per-kind query details the
domain must not know (BM25 columns, vector column names, projections).
"""

from __future__ import annotations

__all__ = [
    "AgentCaseIndexAdapter",
    "AgentSkillIndexAdapter",
    "AtomicFactIndexAdapter",
    "EpisodeIndexAdapter",
    "KnowledgeTopicIndexAdapter",
    "ProfileIndexAdapter",
]

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel

from everos.component.utils.datetime import to_timestamp_ms
from everos.core.observability.logging import get_logger
from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    EpisodeParentType,
    KnowledgeTopic,
    Profile,
)
from everos.memory.ports import (
    Filter,
    MemoryRef,
    OwnerType,
    Page,
    PageResult,
    Scope,
    ScoredKnowledgeTopic,
    SearchHit,
    Stored,
    primary_key,
)

from .protocols import (
    AgentSkillIndexRepository,
    EpisodeIndexRepository,
    IndexRepository,
)

logger = get_logger(__name__)

_VECTOR_COLUMNS = frozenset({"vector", "subject_vector"})

type Row = Mapping[str, Any]


class _SearchableIndex[T]:
    """BM25 + ANN + filtered scan shared by every owner-scoped searchable kind.

    Args:
        repo: The kind's derived-index repository.
        to_stored: The kind's row -> domain conversion.
    """

    def __init__(
        self, repo: IndexRepository[Any], to_stored: Callable[[Row], Stored[T]]
    ) -> None:
        self._repo = repo
        self._to_stored = to_stored

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[T]]:
        rows = await self._repo.sparse_search(
            terms, where, columns=self._repo.schema.BM25_FIELDS, limit=limit
        )
        return [self._hit(row, _bm25_score(row)) for row in rows]

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[T]]:
        rows = await self._repo.dense_search(vector, where, limit=limit)
        return [self._hit(row, _cosine_similarity(row)) for row in rows]

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[T]]:
        if limit is None:
            rows: Sequence[BaseModel | Row] = await self._repo.scan(where)
        else:
            rows = await self._repo.search(where=where, limit=limit)
        return [self._hit(row, None) for row in rows]

    def _hit(self, row: BaseModel | Row, score: float | None) -> SearchHit[T]:
        return SearchHit(stored=self._to_stored(_fields(row)), score=score)

    async def _paginate(self, where: Filter, page: Page) -> PageResult[Stored[T]]:
        rows, total = await self._repo.find_where_paginated(
            where,
            sort_by=page.sort_by,
            descending=page.descending,
            page=page.number,
            page_size=page.size,
        )
        return PageResult(
            items=[self._to_stored(_fields(r)) for r in rows], total=total
        )


class EpisodeIndexAdapter(_SearchableIndex[Episode]):
    """Episode index reads.

    Args:
        repo: Episode derived-index repository.
    """

    def __init__(self, repo: EpisodeIndexRepository[Any]) -> None:
        super().__init__(repo, _stored_episode)
        self._episodes = repo

    async def dense_search_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[Episode]]:
        rows = await self._episodes.dense_search(
            vector, where, limit=limit, vector_field="subject_vector"
        )
        return [self._hit(row, _cosine_similarity(row)) for row in rows]

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[Episode]]:
        rows = await self._episodes.find_by_owner_entries(
            scope.owner_id,
            entry_ids,
            app_id=scope.app_id,
            project_id=scope.project_id,
            skip_vectors=True,
        )
        return [_stored_episode(_fields(r)) for r in rows]

    async def paginate(self, where: Filter, page: Page) -> PageResult[Stored[Episode]]:
        return await self._paginate(where, page)

    async def count(self, scope: Scope, *, parent_type: EpisodeParentType) -> int:
        return await self._episodes.count_by_owner(
            scope.owner_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            parent_type=parent_type,
        )

    async def list_after(
        self,
        scope: Scope,
        after_ts: int,
        *,
        parent_type: EpisodeParentType,
        limit: int | None = None,
    ) -> list[Stored[Episode]]:
        rows = await self._episodes.list_by_owner_after_ts(
            owner_id=scope.owner_id,
            after_ts=after_ts,
            parent_type=parent_type,
            app_id=scope.app_id,
            project_id=scope.project_id,
            columns=_read_columns(self._episodes.schema),
            limit=limit,
        )
        return [_stored_episode(_fields(r)) for r in rows]

    async def orphans(self, where: Filter, *, limit: int) -> list[Stored[Episode]]:
        rows = await self._episodes.find_where(where, limit=limit, skip_vectors=True)
        return [_stored_episode(_fields(r)) for r in rows]


class AtomicFactIndexAdapter(_SearchableIndex[AtomicFact]):
    """Atomic-fact index reads.

    Args:
        repo: Atomic-fact derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        super().__init__(repo, _stored_atomic_fact)


class AgentCaseIndexAdapter(_SearchableIndex[AgentCase]):
    """Agent-case index reads.

    Args:
        repo: Agent-case derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        super().__init__(repo, _stored_agent_case)

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[AgentCase]]:
        rows = await self._repo.find_by_owner_entries(
            scope.owner_id,
            entry_ids,
            app_id=scope.app_id,
            project_id=scope.project_id,
            skip_vectors=True,
        )
        return [_stored_agent_case(_fields(r)) for r in rows]

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[AgentCase]]:
        return await self._paginate(where, page)


class AgentSkillIndexAdapter(_SearchableIndex[AgentSkill]):
    """Agent-skill index reads.

    Args:
        repo: Agent-skill derived-index repository.
    """

    def __init__(self, repo: AgentSkillIndexRepository[Any]) -> None:
        super().__init__(repo, _stored_agent_skill)
        self._skills = repo

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[AgentSkill]]:
        return await self._paginate(where, page)

    async def count(self) -> int:
        return await self._skills.count()

    async def count_in_cluster(self, scope: Scope, cluster_id: str) -> int:
        return await self._skills.count_in_cluster(
            owner_id=scope.owner_id, cluster_id=cluster_id
        )

    async def top_k_in_cluster(
        self, scope: Scope, cluster_id: str, vector: Sequence[float], k: int
    ) -> list[Stored[AgentSkill]]:
        rows = await self._skills.find_topk_relevant_in_cluster(
            owner_id=scope.owner_id,
            cluster_id=cluster_id,
            query_vector=vector,
            top_k=k,
        )
        return [_stored_agent_skill(_fields(r)) for r in rows]


class ProfileIndexAdapter:
    """The one profile row per user.

    Args:
        repo: User-profile derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        self._repo = repo

    async def get(self, scope: Scope) -> Stored[Profile] | None:
        row = await self._repo.find_one_where(
            primary_key(scope.app_id, scope.project_id, scope.owner_id)
        )
        if row is None:
            return None
        fields = _fields(row)
        return Stored(
            ref=MemoryRef(
                id=fields["id"], entry_id=fields["id"], md_path=fields["md_path"]
            ),
            scope=_scope(fields),
            item=Profile.from_stored(
                owner_id=fields["owner_id"],
                summary=fields["summary"],
                timestamp=fields["profile_timestamp_ms"],
                explicit_info=_json_list(fields["explicit_info_json"]),
                implicit_traits=_json_list(fields["implicit_traits_json"]),
            ),
        )


class KnowledgeTopicIndexAdapter:
    """Knowledge-topic index search.

    Topics have no owner, so matches come back as
    :class:`ScoredKnowledgeTopic` rather than an owner-scoped ``Stored``.

    Args:
        repo: Knowledge-topic derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        self._repo = repo

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[ScoredKnowledgeTopic]:
        rows = await self._repo.sparse_search(
            terms, where, columns=self._repo.schema.BM25_FIELDS, limit=limit
        )
        return [_scored_topic(_fields(row), _bm25_score(row)) for row in rows]

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[ScoredKnowledgeTopic]:
        rows = await self._repo.dense_search(vector, where, limit=limit)
        return [_scored_topic(_fields(row), _cosine_similarity(row)) for row in rows]


# ── Row -> domain ────────────────────────────────────────────────────────


def _fields(row: BaseModel | Row) -> Row:
    return row.model_dump(mode="python") if isinstance(row, BaseModel) else row


def _bm25_score(row: Row) -> float | None:
    score = row.get("_score")
    return None if score is None else float(score)


def _cosine_similarity(row: Row) -> float:
    """Both engines report ``_distance = 1 - cos``; a missing one scores 0."""
    distance = row.get("_distance")
    if distance is None:
        return 0.0
    return min(max(1.0 - float(distance), 0.0), 1.0)


def _read_columns(schema: type[BaseModel]) -> list[str]:
    """Every column a domain read needs: no vectors, no BM25 token columns."""
    tokens = set(getattr(schema, "BM25_FIELDS", ()))
    return [
        name
        for name in schema.model_fields
        if name not in _VECTOR_COLUMNS and name not in tokens
    ]


def _scope(fields: Row) -> Scope:
    return Scope(
        app_id=fields["app_id"],
        project_id=fields["project_id"],
        owner_id=fields["owner_id"],
        owner_type=cast(OwnerType, fields["owner_type"]),
    )


def _daily_ref(fields: Row) -> MemoryRef:
    return MemoryRef(
        id=fields["id"], entry_id=fields["entry_id"], md_path=fields["md_path"]
    )


def _stored_episode(fields: Row) -> Stored[Episode]:
    return Stored(
        ref=_daily_ref(fields),
        scope=_scope(fields),
        item=Episode(
            owner_id=fields["owner_id"],
            episode=fields["episode"],
            timestamp=to_timestamp_ms(fields["timestamp"]),
            # Rows written before either column existed hold ``None``.
            subject=fields.get("subject") or "",
            summary=fields.get("summary") or "",
            session_id=fields.get("session_id"),
            sender_ids=list(fields.get("sender_ids") or []),
            parent_id=fields["parent_id"],
            parent_type=fields.get("parent_type") or "memcell",
        ),
    )


def _stored_atomic_fact(fields: Row) -> Stored[AtomicFact]:
    return Stored(
        ref=_daily_ref(fields),
        scope=_scope(fields),
        item=AtomicFact(
            owner_id=fields["owner_id"],
            content=fields["fact"],
            timestamp=to_timestamp_ms(fields["timestamp"]),
            session_id=fields.get("session_id"),
            parent_id=fields["parent_id"],
        ),
    )


def _stored_agent_case(fields: Row) -> Stored[AgentCase]:
    return Stored(
        ref=_daily_ref(fields),
        scope=_scope(fields),
        item=AgentCase(
            owner_id=fields["owner_id"],
            task_intent=fields["task_intent"],
            approach=fields["approach"],
            quality_score=fields["quality_score"],
            key_insight=fields.get("key_insight"),
            timestamp=to_timestamp_ms(fields["timestamp"]),
            session_id=fields["session_id"],
            parent_id=fields["parent_id"],
        ),
    )


def _stored_agent_skill(fields: Row) -> Stored[AgentSkill]:
    return Stored(
        ref=MemoryRef(
            id=fields["id"], entry_id=fields["name"], md_path=fields["md_path"]
        ),
        scope=_scope(fields),
        item=AgentSkill(
            id=fields["id"],
            cluster_id=fields.get("cluster_id") or "",
            name=fields["name"],
            description=fields["description"],
            content=fields["content"],
            confidence=fields["confidence"],
            maturity_score=fields["maturity_score"],
            source_case_ids=list(fields.get("source_case_ids") or []),
        ),
    )


def _scored_topic(fields: Row, score: float | None) -> ScoredKnowledgeTopic:
    return ScoredKnowledgeTopic(
        topic=KnowledgeTopic(
            topic_id=fields["id"],
            doc_id=fields["doc_id"],
            app_id=fields["app_id"],
            project_id=fields["project_id"],
            category_id=fields["category_id"],
            topic_name=fields["topic_name"],
            topic_path=fields["topic_path"],
            depth=fields["depth"],
            summary=fields["summary"],
            parent_topic_id=fields.get("parent_node_id") or None,
            content_labels=list(fields.get("content_labels") or []),
            created_at=fields["created_at"],
            updated_at=fields["updated_at"],
        ),
        md_path=fields["md_path"],
        score=score,
    )


def _json_list(text: str) -> list[Any]:
    """Decode a json-encoded profile bucket; ``[]`` when empty or malformed.

    A stale encoding must not break a read; the cascade rewrites the column
    on its next reconcile.
    """
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("profile_json_decode_failed", payload_head=text[:80])
        return []
    return decoded if isinstance(decoded, list) else []
