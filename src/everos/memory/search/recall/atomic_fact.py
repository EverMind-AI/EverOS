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

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.memory.ports import AtomicFactIndex, Filter, IndexHit, all_of, one_of
from everos.memory.search._candidates import FactCandidate

from .base import (
    RecallerDeps,
    cosine_score_from_distance,
    row_to_candidate,
)

_NOISE_COLUMNS = frozenset(
    {"vector", "_distance", "_score", "created_at", "updated_at"}
)


class AtomicFactRecaller:
    """BM25 + vector recall over the atomic-fact index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The atomic-fact index port.
    """

    kind: ClassVar[str] = "atomic_fact"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "fact"

    def __init__(self, deps: RecallerDeps, index: AtomicFactIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> list[Candidate]:
        """BM25 recall via OR-mode BooleanQuery (see EpisodeRecaller docstring)."""
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return []
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return [
            row_to_candidate(h, source="keyword", score=h.score or 0.0) for h in hits
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[Candidate]:
        """Cosine ANN recall over the atomic_fact table.

        Args:
            vector: Query embedding vector; empty returns no results.
            where: LanceDB SQL filter clause scoping the search.
            limit: Maximum number of candidates to return.

        Returns:
            Candidates ranked by cosine similarity (descending).
        """
        if not vector:
            return []
        hits = await self._index.dense_search(vector, where, limit=limit)
        return [
            row_to_candidate(
                h,
                source="vector",
                score=cosine_score_from_distance(h.fields.get("_distance")),
            )
            for h in hits
        ]

    async def facts_for_episodes(
        self,
        ep_to_parents: Mapping[str, Sequence[str]],
        where: Filter,
        *,
        per_episode: int,
        query_vector: Sequence[float] | None = None,
    ) -> dict[str, list[FactCandidate]]:
        """Pull facts for a set of episodes, bucketed by episode id.

        ``ep_to_parents`` maps the candidate episode's LanceDB id to a
        list of possible fact parent_id values (entry_id for post-1.5
        facts, memcell_id for pre-1.5 facts). Facts are queried by
        ``parent_id IN (all_unique_parent_ids)`` and re-bucketed under
        every episode that claims each parent_id — two episodes sharing
        a parent_id each get a copy of that parent's facts.

        When ``query_vector`` is provided, the LanceDB query layers
        cosine ANN on top of the ``parent_id IN (...)`` filter, so each
        fact lands with a real query-fact relevance score.
        Without ``query_vector`` we fall back to a flat scan, in which
        case every fact ships with ``score=0.0`` — the caller is
        responsible for not consuming the score in that mode.
        """
        if not ep_to_parents:
            return {}

        parent_to_eps = _build_parent_to_episode_map(ep_to_parents)
        if not parent_to_eps:
            return {}

        hits = await self._query_facts_for_parents(
            parent_to_eps, where, per_episode=per_episode, query_vector=query_vector
        )

        # Bucket hits by episode and cap each bucket.
        buckets: dict[str, list[FactCandidate]] = defaultdict(list)
        for hit in hits:
            fact_parent_id = hit.fields.get("parent_id")
            if not isinstance(fact_parent_id, str):
                continue
            metadata = {
                k: v
                for k, v in hit.fields.items()
                if k not in _NOISE_COLUMNS and k != "id"
            }
            score = (
                cosine_score_from_distance(hit.fields.get("_distance"))
                if query_vector
                else 0.0
            )
            for ep_id in parent_to_eps.get(fact_parent_id, ()):
                buckets[ep_id].append(
                    FactCandidate(
                        id=hit.id,
                        parent_episode_id=ep_id,
                        score=score,
                        metadata=metadata,
                    )
                )
        # With query_vector the rows arrive sorted by cosine ascending
        # (closest first) so slicing keeps the most relevant facts.
        return {ep_id: bucket[:per_episode] for ep_id, bucket in buckets.items()}

    async def _query_facts_for_parents(
        self,
        parent_to_eps: dict[str, list[str]],
        where: Filter,
        *,
        per_episode: int,
        query_vector: Sequence[float] | None,
    ) -> list[IndexHit]:
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
