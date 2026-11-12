"""Stored-record builders and a scripted index for the search-layer tests.

Recall tests drive a recaller through its ``*Index`` port; the port already
returns domain records (``SearchHit``), so a scripted index that hands back
prepared hits is all a recall test needs. The adapter's row -> domain
conversion is covered by the index adapter tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from everos.memory.models import Episode
from everos.memory.ports import (
    Filter,
    MemoryRef,
    OwnerType,
    Scope,
    SearchHit,
    Stored,
)
from everos.memory.search.recall import RecallResult
from everos.memory.search.recall.base import from_hits
from tests.fakes._filter import matches

TS_MS = 1_767_225_600_000
"""2026-01-01T00:00:00Z."""


def scope(owner_id: str = "alice", owner_type: OwnerType = "user") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type=owner_type
    )


def _ref(owner_id: str, entry_id: str, md_path: str = "x.md") -> MemoryRef:
    return MemoryRef(id=f"{owner_id}_{entry_id}", entry_id=entry_id, md_path=md_path)


def stored_episode(
    entry_id: str,
    *,
    owner_id: str = "alice",
    owner_type: OwnerType = "user",
    parent_id: str = "mc_1",
    parent_type: str = "memcell",
    session_id: str = "sess_a",
    timestamp: int = TS_MS,
) -> Stored[Episode]:
    return Stored(
        ref=_ref(owner_id, entry_id, md_path=f"users/{owner_id}/episodes/e.md"),
        scope=scope(owner_id, owner_type),
        item=Episode(
            owner_id=owner_id,
            episode=f"body {entry_id}",
            timestamp=timestamp,
            subject=f"subject {entry_id}",
            summary=f"summary {entry_id}",
            session_id=session_id,
            sender_ids=[owner_id],
            parent_id=parent_id,
            parent_type=parent_type,  # type: ignore[arg-type]
        ),
    )


def hit[T](stored: Stored[T], score: float | None = None) -> SearchHit[T]:
    return SearchHit(stored=stored, score=score)


class ScriptedIndex:
    """An ``*Index`` whose search methods return prepared hits and record calls.

    Args:
        sparse: Hits for ``sparse_search``.
        dense: Hits for ``dense_search``.
        visual: Hits for ``visual_search`` (a picture's own embedding).
        subject: Hits for ``dense_search_subject``.
        found: Hits for ``find``.
        filter_found: Evaluate ``find``'s filter against each record, as
            the real index does, instead of returning ``found`` as is.
    """

    def __init__(
        self,
        *,
        sparse: Sequence[Any] = (),
        dense: Sequence[Any] = (),
        visual: Sequence[Any] = (),
        subject: Sequence[Any] = (),
        found: Sequence[Any] = (),
        filter_found: bool = False,
    ) -> None:
        self._sparse = list(sparse)
        self._dense = list(dense)
        self._visual = list(visual)
        self._subject = list(subject)
        self._found = list(found)
        self._filter_found = filter_found
        self.calls: list[tuple[str, Any, Filter, int | None]] = []

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[Any]:
        self.calls.append(("sparse", list(terms), where, limit))
        return self._sparse[:limit]

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Any]:
        self.calls.append(("dense", list(vector), where, limit))
        return self._dense[:limit]

    async def visual_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Any]:
        self.calls.append(("visual", list(vector), where, limit))
        return self._visual[:limit]

    async def dense_search_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Any]:
        self.calls.append(("subject", list(vector), where, limit))
        return self._subject[:limit]

    async def find(self, where: Filter, *, limit: int | None = None) -> list[Any]:
        self.calls.append(("find", None, where, limit))
        found = [
            h
            for h in self._found
            if not self._filter_found or matches(where, _fields(h))
        ]
        return found if limit is None else found[:limit]


def _fields(h: SearchHit[Any]) -> dict[str, Any]:
    """The columns a filter may name, read off a stored record."""
    stored = h.stored
    return {
        **stored.item.model_dump(),
        "id": stored.ref.id,
        "entry_id": stored.ref.entry_id,
        "md_path": stored.ref.md_path,
        "owner_id": stored.scope.owner_id,
        "owner_type": stored.scope.owner_type,
        "app_id": stored.scope.app_id,
        "project_id": stored.scope.project_id,
        "deprecated_by": None,
    }


def recalled(*hits: SearchHit[Any], source: str = "vector") -> RecallResult[Any]:
    """What a recaller returns for ``hits`` -- the production mapping."""
    return from_hits(hits, source=source)  # type: ignore[arg-type]
