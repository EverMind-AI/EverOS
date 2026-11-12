"""In-memory implementations of the read-side index ports.

Indexes are projections the cascade maintains; these doubles are seeded
directly by the test (``seed``) and never see what the stores write, which
is the same lag the real system has.
"""

from __future__ import annotations

__all__ = [
    "InMemoryEpisodeIndex",
    "InMemoryGroupProfileIndex",
    "InMemoryImageMemoryIndex",
    "InMemoryProfileIndex",
]

from collections.abc import Mapping, Sequence
from typing import Any

from everos.core.utils.datetime import from_timestamp_ms
from everos.memory.models import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    Profile,
)
from everos.memory.ports import Filter, Page, PageResult, Scope, SearchHit, Stored

from ._filter import matches


def _same_space(a: Scope, b: Scope) -> bool:
    return (a.owner_id, a.app_id, a.project_id) == (b.owner_id, b.app_id, b.project_id)


class _Unsearchable:
    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[Any]]:
        raise NotImplementedError

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[Any]]:
        raise NotImplementedError

    async def visual_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[Any]]:
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
    ) -> list[SearchHit[Episode]]:
        raise NotImplementedError

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[Episode]]:
        raise NotImplementedError

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


class InMemoryImageMemoryIndex(_Unsearchable):
    """Seeded picture memories; search routes raise until a test needs one."""

    def __init__(self) -> None:
        self.rows: list[Stored[ImageMemory]] = []

    def seed(self, *rows: Stored[ImageMemory]) -> None:
        self.rows.extend(rows)

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[ImageMemory]]:
        hits = [
            SearchHit(stored=row, score=None)
            for row in self.rows
            if matches(where, _image_fields(row))
        ]
        return hits if limit is None else hits[:limit]

    async def find_by_image_ids(
        self, scope: Scope, image_ids: Sequence[str]
    ) -> list[Stored[ImageMemory]]:
        wanted = set(image_ids)
        return [
            r
            for r in self.rows
            if _same_space(r.scope, scope)
            and r.item.image_id in wanted
            and r.item.superseded_by is None
        ]

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[ImageMemory]]:
        wanted = set(entry_ids)
        return [
            r
            for r in self.rows
            if _same_space(r.scope, scope) and r.ref.entry_id in wanted
        ]

    async def find_by_parent(
        self, *, parent_type: str, parent_id: str, limit: int = 100
    ) -> list[Stored[ImageMemory]]:
        return [
            r
            for r in self.rows
            if r.item.parent_id == parent_id and r.item.parent_type == parent_type
        ][:limit]


def _image_fields(row: Stored[ImageMemory]) -> Mapping[str, Any]:
    return {
        **row.item.model_dump(),
        "id": row.ref.id,
        "entry_id": row.ref.entry_id,
        "md_path": row.ref.md_path,
        "app_id": row.scope.app_id,
        "project_id": row.scope.project_id,
        "owner_type": row.scope.owner_type,
        "deprecated_by": None,
    }


class InMemoryGroupProfileIndex:
    """Seeded group-profile rows; marks (``deprecated_by``) kept per entry id.

    Mirrors the adapter: filters see the row's columns under their domain
    names (times as datetimes, ``source_memcell_ids`` a list), ``find``
    returns newest ``observed_at`` first, ``find_by_entry_ids`` ignores the
    marks, and every item comes back with the status its mark gives.
    ``sparse_search`` scores by the number of query terms a statement holds.
    """

    def __init__(self) -> None:
        self.rows: list[Stored[GroupProfileItem]] = []
        self.marks: dict[str, str] = {}

    def seed(self, *rows: Stored[GroupProfileItem]) -> None:
        self.rows.extend(rows)

    def _read(self, row: Stored[GroupProfileItem]) -> Stored[GroupProfileItem]:
        item = row.item.read_back(
            item_id=row.ref.entry_id, mark=self.marks.get(row.ref.entry_id)
        )
        return Stored(ref=row.ref, scope=row.scope, item=item)

    def _fields(self, row: Stored[GroupProfileItem]) -> Mapping[str, Any]:
        item = row.item
        return {
            **item.model_dump(exclude={"sources"}),
            "source_memcell_ids": item.source_memcell_ids,
            "source_start": from_timestamp_ms(item.source_start),
            "source_end": from_timestamp_ms(item.source_end),
            "observed_at": from_timestamp_ms(item.observed_at),
            "id": row.ref.id,
            "entry_id": row.ref.entry_id,
            "md_path": row.ref.md_path,
            "app_id": row.scope.app_id,
            "project_id": row.scope.project_id,
            "owner_type": row.scope.owner_type,
            "deprecated_by": self.marks.get(row.ref.entry_id),
        }

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[Stored[GroupProfileItem]]:
        found = sorted(
            (row for row in self.rows if matches(where, self._fields(row))),
            key=lambda row: (row.item.observed_at, row.ref.entry_id),
            reverse=True,
        )
        read = [self._read(row) for row in found]
        return read if limit is None else read[:limit]

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[GroupProfileItem]]:
        wanted = {t.lower() for t in terms}
        scored: list[tuple[float, Stored[GroupProfileItem]]] = []
        for row in self.rows:
            if not matches(where, self._fields(row)):
                continue
            text = f"{row.item.statement} {row.item.name} {row.item.summary}"
            score = float(sum(1 for word in text.lower().split() if word in wanted))
            if score:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchHit(stored=self._read(row), score=score)
            for score, row in scored[:limit]
        ]

    async def find_by_entry_ids(
        self, scope: Scope, entry_ids: Sequence[str]
    ) -> list[Stored[GroupProfileItem]]:
        wanted = set(entry_ids)
        return [
            self._read(r)
            for r in self.rows
            if _same_space(r.scope, scope) and r.ref.entry_id in wanted
        ]


class InMemoryProfileIndex:
    """Seeded profiles by scope; member-in-group ones by ``(group scope, member)``."""

    def __init__(self) -> None:
        self.rows: dict[Scope, Stored[Profile]] = {}
        self.members: dict[tuple[Scope, str], Stored[Profile]] = {}

    async def get(self, scope: Scope) -> Stored[Profile] | None:
        return self.rows.get(scope)

    async def member(self, scope: Scope, member_id: str) -> Stored[Profile] | None:
        return self.members.get((scope, member_id))
