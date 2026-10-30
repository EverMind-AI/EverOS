"""Episode recaller -- BM25 + cosine ANN over the episode index."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everos.memory.models import Episode
from everos.memory.ports import EpisodeIndex, Filter, SearchHit, Stored, all_of, one_of

from .base import RecallerDeps, RecallResult, from_hits, to_candidate


class EpisodeRecaller:
    """BM25 + vector recall over the episode index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The episode index port.
    """

    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"

    def __init__(self, deps: RecallerDeps, index: EpisodeIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        """BM25 recall via OR-mode BooleanQuery.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token (typically the partition owner's own name on
        owner-scoped corpora) cannot poison the entire query.
        Mirrors enterprise's ``bool.should + minimum_should_match=1``
        ES design.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return from_hits(hits, source="keyword")

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return from_hits(hits, source="vector")

    async def sparse_recall_as_child(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        """Sparse recall returning episodes as MaxSim child candidates."""
        return _as_children(await self.sparse_recall(query, where, limit=limit))

    async def dense_recall_as_child(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        """Dense recall (body vector ANN) returning as MaxSim children."""
        return _as_children(await self.dense_recall(vector, where, limit=limit))

    async def dense_recall_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        """ANN over the subject embedding; episodes without one never match."""
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search_subject(vector, where, limit=limit)
        return from_hits(hits, source="vector")

    async def dense_recall_subject_as_child(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[Episode]:
        """Subject-vector ANN returning as MaxSim children."""
        return _as_children(await self.dense_recall_subject(vector, where, limit=limit))

    async def fetch_all_for_owner(self, where: Filter) -> RecallResult[Episode]:
        """Every episode in ``where``, keyed by ``entry_id``.

        Cluster membership matching in ``acluster_retrieve`` compares
        ``Candidate.id`` against ``Cluster.members``, which are episode
        entry ids. No ``limit``: the whole owner partition is needed.
        """
        return _by_entry_id(await self._index.find(where), score=0.0)

    async def fetch_parents(
        self, entry_ids: Sequence[str], where: Filter
    ) -> RecallResult[Episode]:
        """Episodes by ``entry_id``, keyed by it -- the MaxSim parent fetch.

        A child's ``parent_id`` is its episode's entry id whether the episode
        came from a memcell or a cluster merge, so parents are addressed by it.
        """
        if not entry_ids:
            return RecallResult.empty()
        full_where = all_of(where, one_of("entry_id", list(entry_ids)))
        hits = await self._index.find(full_where, limit=len(entry_ids))
        return _by_entry_id(hits, score=0.0)

    async def fetch_by_entry_ids(
        self, entry_ids: Sequence[str], where: Filter
    ) -> RecallResult[Episode]:
        """Episodes by ``entry_id``, keyed by global id, unscored."""
        if not entry_ids:
            return RecallResult.empty()
        full_where = all_of(where, one_of("entry_id", list(entry_ids)))
        hits = await self._index.find(full_where, limit=len(entry_ids))
        return from_hits(hits, source="vector")


def _as_children(result: RecallResult[Episode]) -> RecallResult[Episode]:
    """Tag each candidate with ``parent_id = entry_id`` for MaxSim grouping."""
    return RecallResult(
        candidates=[
            c.model_copy(
                update={"metadata": {**c.metadata, "parent_id": c.metadata["entry_id"]}}
            )
            for c in result.candidates
        ],
        stored_by_id=result.stored_by_id,
    )


def _by_entry_id(
    hits: Sequence[SearchHit[Episode]], *, score: float
) -> RecallResult[Episode]:
    candidates = []
    stored_by_id: dict[str, Stored[Episode]] = {}
    for hit in hits:
        entry_id = hit.stored.ref.entry_id
        candidates.append(
            to_candidate(hit.stored, score=score, source="vector", key=entry_id)
        )
        stored_by_id[entry_id] = hit.stored
    return RecallResult(candidates=candidates, stored_by_id=stored_by_id)
