"""AtomicFact recaller — BM25 over ``fact_tokens`` + cosine ANN + parent linkage.

Beyond the standard sparse / dense pair the recaller exposes
:meth:`facts_for_episodes`, which the HYBRID pipeline calls to attach
atomic facts to their parent episodes (``episode_to_facts`` fed into
the fact eviction pass).

Episode-fact linkage uses a **dual parent_id strategy**:
- New facts (post-1.5): ``parent_id = episode_entry_id``.
- Old facts (pre-1.5): ``parent_id = memcell_id``.
The caller hands in an ``episode_id → [parent_id, ...]`` map; we query
facts by ``parent_id IN (all_parent_ids)`` and regroup by episode using
the inverse map, so both old and new facts are surfaced without backfill.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import ClassVar

from everos.memory.models import AtomicFact
from everos.memory.ports import (
    AtomicFactIndex,
    Filter,
    SearchHit,
    Stored,
    all_of,
    one_of,
)
from everos.memory.search._candidates import FactCandidate

from .base import RecallerDeps, RecallResult, from_hits


@dataclasses.dataclass(frozen=True)
class FactsByEpisode:
    """Atomic facts grouped under the episodes that claim them.

    Attributes:
        buckets: Episode candidate id -> its facts, most relevant first.
        stored_by_id: The stored fact behind each ``FactCandidate.id``.
    """

    buckets: dict[str, list[FactCandidate]]
    stored_by_id: Mapping[str, Stored[AtomicFact]]


class AtomicFactRecaller:
    """BM25 + vector recall over the atomic-fact index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The atomic-fact index port.
    """

    kind: ClassVar[str] = "atomic_fact"
    everalgo_memory_type: ClassVar[str] = "episodic"

    def __init__(self, deps: RecallerDeps, index: AtomicFactIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[AtomicFact]:
        """BM25 recall via OR-mode BooleanQuery (see EpisodeRecaller docstring)."""
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return from_hits(hits, source="keyword")

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[AtomicFact]:
        """Cosine ANN recall over the atomic-fact index.

        Args:
            vector: Query embedding vector; empty returns no results.
            where: Filter scoping the search.
            limit: Maximum number of candidates to return.

        Returns:
            Candidates ranked by cosine similarity (descending).
        """
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return from_hits(hits, source="vector")

    async def facts_for_episodes(
        self,
        ep_to_parents: Mapping[str, Sequence[str]],
        where: Filter,
        *,
        per_episode: int,
        query_vector: Sequence[float] | None = None,
    ) -> FactsByEpisode:
        """Pull facts for a set of episodes, bucketed by episode id.

        ``ep_to_parents`` maps the candidate episode's id to a list of
        possible fact parent_id values (entry_id for post-1.5 facts,
        memcell_id for pre-1.5 facts). Facts are queried by
        ``parent_id IN (all_unique_parent_ids)`` and re-bucketed under
        every episode that claims each parent_id -- two episodes sharing
        a parent_id each get a copy of that parent's facts.

        When ``query_vector`` is provided, the query layers cosine ANN on
        top of the ``parent_id IN (...)`` filter, so each fact lands with a
        real query-fact relevance score. Without it we fall back to a flat
        scan, in which case every fact ships with ``score=0.0`` -- the
        caller is responsible for not consuming the score in that mode.
        """
        if not ep_to_parents:
            return FactsByEpisode(buckets={}, stored_by_id={})

        parent_to_eps = _build_parent_to_episode_map(ep_to_parents)
        if not parent_to_eps:
            return FactsByEpisode(buckets={}, stored_by_id={})

        hits = await self._query_facts_for_parents(
            parent_to_eps, where, per_episode=per_episode, query_vector=query_vector
        )

        buckets: dict[str, list[FactCandidate]] = defaultdict(list)
        stored_by_id: dict[str, Stored[AtomicFact]] = {}
        for hit in hits:
            fact = hit.stored
            stored_by_id[fact.ref.id] = fact
            score = (hit.score or 0.0) if query_vector else 0.0
            for ep_id in parent_to_eps.get(fact.item.parent_id, ()):
                buckets[ep_id].append(
                    FactCandidate(
                        id=fact.ref.id,
                        parent_episode_id=ep_id,
                        score=score,
                        metadata={"parent_id": fact.item.parent_id},
                    )
                )
        # With query_vector the hits arrive closest first, so slicing keeps
        # the most relevant facts.
        return FactsByEpisode(
            buckets={ep_id: bucket[:per_episode] for ep_id, bucket in buckets.items()},
            stored_by_id=stored_by_id,
        )

    async def _query_facts_for_parents(
        self,
        parent_to_eps: dict[str, list[str]],
        where: Filter,
        *,
        per_episode: int,
        query_vector: Sequence[float] | None,
    ) -> list[SearchHit[AtomicFact]]:
        """Query the facts whose ``parent_id`` is one of the given parents."""
        full_where = all_of(where, one_of("parent_id", list(parent_to_eps)))
        limit = per_episode * max(len(parent_to_eps), 1)
        if query_vector:
            return await self._index.dense_search(query_vector, full_where, limit=limit)
        return await self._index.find(full_where, limit=limit)


def _build_parent_to_episode_map(
    ep_to_parents: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Invert ep-to-parents map to a parent-to-episodes map."""
    parent_to_eps: dict[str, list[str]] = defaultdict(list)
    for ep_id, parent_ids in ep_to_parents.items():
        for pid in parent_ids:
            if pid:
                parent_to_eps[pid].append(ep_id)
    return parent_to_eps
