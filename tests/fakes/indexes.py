"""In-memory implementations of the read-side index ports.

Indexes are projections the cascade maintains; these doubles are seeded
directly by the test (``seed``) and never see what the stores write, which
is the same lag the real system has.
"""

from __future__ import annotations

__all__ = [
    "InMemoryAgentCaseIndex",
    "InMemoryAgentSkillIndex",
    "InMemoryAtomicFactIndex",
    "InMemoryEpisodeIndex",
    "InMemoryProfileIndex",
]

from collections.abc import Mapping, Sequence
from typing import Any

from everos.memory.models import AgentCase, AgentSkill, Episode, Profile
from everos.memory.ports import Filter, IndexHit, Page, PageResult, Scope, Stored

from ._filter import matches


def _same_space(a: Scope, b: Scope) -> bool:
    return (a.owner_id, a.app_id, a.project_id) == (b.owner_id, b.app_id, b.project_id)


class _Unsearchable:
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[IndexHit]:
        raise NotImplementedError

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[IndexHit]:
        raise NotImplementedError


class InMemoryEpisodeIndex(_Unsearchable):
    """Seeded episodes; ``deprecated_by`` is tracked per entry id."""

    def __init__(self) -> None:
        self.rows: list[Stored[Episode]] = []
        self.deprecated_by: dict[str, str] = {}

    def seed(self, *rows: Stored[Episode]) -> None:
        self.rows.extend(rows)

    def _fields(self, row: Stored[Episode]) -> Mapping[str, Any]:
        return {
            **row.item.model_dump(),
            "id": row.ref.id,
            "entry_id": row.ref.entry_id,
            "md_path": row.ref.md_path,
            "app_id": row.scope.app_id,
            "project_id": row.scope.project_id,
            "owner_type": row.scope.owner_type,
            "deprecated_by": self.deprecated_by.get(row.ref.entry_id),
        }

    async def dense_search_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[IndexHit]:
        raise NotImplementedError

    async def find(self, where: Filter, *, limit: int | None = None) -> list[IndexHit]:
        raise NotImplementedError

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[Episode]]:
        wanted = set(entry_ids)
        return [
            r
            for r in self.rows
            if _same_space(r.scope, scope) and r.ref.entry_id in wanted
        ]

    async def paginate(self, where: Filter, page: Page) -> PageResult[Stored[Episode]]:
        raise NotImplementedError

    async def count(self, scope: Scope, *, parent_type: str) -> int:
        return sum(
            1
            for r in self.rows
            if _same_space(r.scope, scope) and r.item.parent_type == parent_type
        )

    async def list_after(
        self,
        scope: Scope,
        after_ts: int,
        *,
        parent_type: str,
        limit: int | None = None,
    ) -> list[Stored[Episode]]:
        found = [
            r
            for r in self.rows
            if _same_space(r.scope, scope)
            and r.item.parent_type == parent_type
            and r.item.timestamp > after_ts
        ]
        return found if limit is None else found[:limit]

    async def orphans(self, where: Filter, *, limit: int) -> list[Stored[Episode]]:
        return [r for r in self.rows if matches(where, self._fields(r))][:limit]


class InMemoryAtomicFactIndex(_Unsearchable):
    """Seeded atomic-fact rows as column dicts."""

    def __init__(self) -> None:
        self.rows: list[Mapping[str, Any]] = []

    def seed(self, *rows: Mapping[str, Any]) -> None:
        self.rows.extend(rows)

    async def find(self, where: Filter, *, limit: int | None = None) -> list[IndexHit]:
        hits = [
            IndexHit(id=row["id"], score=None, fields=row)
            for row in self.rows
            if matches(where, row)
        ]
        return hits if limit is None else hits[:limit]


class InMemoryAgentCaseIndex(_Unsearchable):
    """Seeded agent cases."""

    def __init__(self) -> None:
        self.rows: list[Stored[AgentCase]] = []

    def seed(self, *rows: Stored[AgentCase]) -> None:
        self.rows.extend(rows)

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[AgentCase]]:
        wanted = set(entry_ids)
        return [
            r
            for r in self.rows
            if _same_space(r.scope, scope) and r.ref.entry_id in wanted
        ]

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[AgentCase]]:
        raise NotImplementedError


class InMemoryAgentSkillIndex(_Unsearchable):
    """Seeded skills; seed order is the relevance ranking ``top_k`` returns."""

    def __init__(self) -> None:
        self.rows: list[Stored[AgentSkill]] = []
        self.top_k_calls: list[tuple[Scope, str, Sequence[float], int]] = []

    def seed(self, *rows: Stored[AgentSkill]) -> None:
        self.rows.extend(rows)

    async def find(self, where: Filter, *, limit: int | None = None) -> list[IndexHit]:
        raise NotImplementedError

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[AgentSkill]]:
        raise NotImplementedError

    async def count(self) -> int:
        return len(self.rows)

    async def count_in_cluster(self, scope: Scope, cluster_id: str) -> int:
        return len(self._in_cluster(scope, cluster_id))

    async def top_k_in_cluster(
        self, scope: Scope, cluster_id: str, vector: Sequence[float], k: int
    ) -> list[Stored[AgentSkill]]:
        self.top_k_calls.append((scope, cluster_id, vector, k))
        return self._in_cluster(scope, cluster_id)[:k]

    def _in_cluster(self, scope: Scope, cluster_id: str) -> list[Stored[AgentSkill]]:
        return [
            r
            for r in self.rows
            if _same_space(r.scope, scope) and r.item.cluster_id == cluster_id
        ]


class InMemoryProfileIndex:
    """Seeded profiles by scope."""

    def __init__(self) -> None:
        self.rows: dict[Scope, Stored[Profile]] = {}

    async def get(self, scope: Scope) -> Stored[Profile] | None:
        return self.rows.get(scope)
