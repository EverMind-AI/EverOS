"""Stored-record builders and a scripted index for the search-layer tests.

Recall tests drive a recaller through its ``*Index`` port; the port already
returns domain records (``SearchHit``), so a scripted index that hands back
prepared hits is all a recall test needs. The adapter's row -> domain
conversion is covered by the index adapter tests.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    KnowledgeTopic,
)
from everos.memory.ports import (
    Filter,
    MemoryRef,
    OwnerType,
    Scope,
    ScoredKnowledgeTopic,
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


def stored_fact(
    entry_id: str, *, parent_id: str, owner_id: str = "alice", md_path: str = "f.md"
) -> Stored[AtomicFact]:
    return Stored(
        ref=_ref(owner_id, entry_id, md_path=md_path),
        scope=scope(owner_id),
        item=AtomicFact(
            owner_id=owner_id,
            content=f"fact {entry_id}",
            timestamp=TS_MS,
            parent_id=parent_id,
        ),
    )


def stored_case(
    entry_id: str,
    *,
    owner_id: str = "agent_1",
    task_intent: str = "fix the build",
    approach: str = "read the log",
    quality_score: float = 0.8,
) -> Stored[AgentCase]:
    return Stored(
        ref=_ref(owner_id, entry_id),
        scope=scope(owner_id, "agent"),
        item=AgentCase(
            owner_id=owner_id,
            task_intent=task_intent,
            approach=approach,
            quality_score=quality_score,
            key_insight=None,
            timestamp=TS_MS,
            session_id="sess_a",
            parent_id="mc_1",
        ),
    )


def stored_skill(
    skill_id: str,
    *,
    owner_id: str = "agent_1",
    name: str = "debugging",
    description: str = "find the bug",
    source_case_ids: Sequence[str] = (),
) -> Stored[AgentSkill]:
    return Stored(
        ref=MemoryRef(id=skill_id, entry_id=name, md_path="skill_x/SKILL.md"),
        scope=scope(owner_id, "agent"),
        item=AgentSkill(
            id=skill_id,
            name=name,
            description=description,
            content="steps",
            confidence=0.7,
            maturity_score=0.6,
            source_case_ids=list(source_case_ids),
        ),
    )


def topic(topic_id: str, *, category_id: str = "Outdoors") -> KnowledgeTopic:
    stamp = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    return KnowledgeTopic(
        topic_id=topic_id,
        doc_id="d_1",
        app_id="default",
        project_id="default",
        category_id=category_id,
        topic_name=f"name {topic_id}",
        topic_path=f"Doc > {topic_id}",
        depth=1,
        summary=f"summary {topic_id}",
        created_at=stamp,
        updated_at=stamp,
    )


def hit[T](stored: Stored[T], score: float | None = None) -> SearchHit[T]:
    return SearchHit(stored=stored, score=score)


def topic_hit(t: KnowledgeTopic, score: float | None = None) -> ScoredKnowledgeTopic:
    return ScoredKnowledgeTopic(
        topic=t, md_path=f"knowledge/{t.topic_id}.md", score=score
    )


class ScriptedIndex:
    """An ``*Index`` whose search methods return prepared hits and record calls.

    Args:
        sparse: Hits for ``sparse_search``.
        dense: Hits for ``dense_search``.
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
        subject: Sequence[Any] = (),
        found: Sequence[Any] = (),
        filter_found: bool = False,
    ) -> None:
        self._sparse = list(sparse)
        self._dense = list(dense)
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
