"""AtomicFact recaller — BM25 over ``fact_tokens`` + cosine ANN + parent linkage.

Beyond the standard sparse / dense pair the recaller exposes
:meth:`facts_for_episodes`, which the HYBRID pipeline calls to attach
atomic facts to their parent episodes (``episode_to_facts`` fed into
the fact eviction pass).

Episode-fact linkage is **indirect through the shared memcell parent**:
both kinds are written with ``parent_id = memcell_id`` by the cascade.
The caller hands in an ``episode_id → memcell_id`` map; we query facts
by ``parent_id IN (memcell_ids)`` and regroup by episode using the
inverse map, so one fact bucket-shows under every episode that shares
the source memcell.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import ClassVar

from everalgo.types import Candidate, FactCandidate

from everos.infra.persistence.lancedb import AtomicFact, get_table

from .base import (
    RecallerDeps,
    build_or_query,
    cosine_score_from_distance,
    row_to_candidate,
)

_NOISE_COLUMNS = frozenset(
    {"vector", "_distance", "_score", "created_at", "updated_at"}
)


class AtomicFactRecaller:
    """BM25 + vector recall over the LanceDB ``atomic_fact`` table."""

    kind: ClassVar[str] = "atomic_fact"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "fact"

    def __init__(self, deps: RecallerDeps) -> None:
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        """BM25 recall via OR-mode BooleanQuery (see EpisodeRecaller docstring)."""
        bq = build_or_query(
            self._deps.tokenizer, query, column=AtomicFact.BM25_FIELDS[0]
        )
        if bq is None:
            return []
        table = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
        rows = (
            await table.query().nearest_to_text(bq).where(where).limit(limit).to_list()
        )
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        if not vector:
            return []
        table = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
        rows = (
            await table.query()
            .nearest_to(list(vector))
            .distance_type("cosine")
            .where(where)
            .limit(limit)
            .to_list()
        )
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]

    async def facts_for_episodes(
        self,
        ep_to_memcell: Mapping[str, str],
        where: str,
        *,
        per_episode: int,
        query_vector: Sequence[float] | None = None,
    ) -> dict[str, list[FactCandidate]]:
        """Pull facts for a set of episodes, bucketed by episode id.

        ``ep_to_memcell`` maps the candidate episode's LanceDB id to the
        source memcell id (read off ``episode.parent_id`` by the
        caller). Facts are queried by their own ``parent_id`` against
        the deduped memcell set, then re-bucketed under every episode
        that shares each memcell — two episodes pulled from the same
        memcell each get a copy of that memcell's facts.

        When ``query_vector`` is provided, the LanceDB query layers
        cosine ANN on top of the ``parent_id IN (...)`` filter, so each
        fact lands with a real query-fact relevance score.
        Without ``query_vector`` we fall back to a flat scan, in which
        case every fact ships with ``score=0.0`` — the caller is
        responsible for not consuming the score in that mode.
        """
        if not ep_to_memcell:
            return {}

        memcell_to_eps: dict[str, list[str]] = defaultdict(list)
        for ep_id, mc_id in ep_to_memcell.items():
            if mc_id:
                memcell_to_eps[mc_id].append(ep_id)
        if not memcell_to_eps:
            return {}

        quoted = ", ".join(f"'{_q(mc_id)}'" for mc_id in memcell_to_eps)
        clause = f"parent_id IN ({quoted})"
        full_where = f"({where}) AND ({clause})"
        limit = per_episode * max(len(memcell_to_eps), 1)
        table = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
        if query_vector:
            rows = (
                await table.query()
                .nearest_to(list(query_vector))
                .distance_type("cosine")
                .where(full_where)
                .limit(limit)
                .to_list()
            )
        else:
            rows = await table.query().where(full_where).limit(limit).to_list()
        buckets: dict[str, list[FactCandidate]] = defaultdict(list)
        for r in rows:
            mc_id = r.get("parent_id")
            fid = r.get("id")
            if not isinstance(mc_id, str) or not isinstance(fid, str):
                continue
            metadata = {
                k: v for k, v in r.items() if k not in _NOISE_COLUMNS and k != "id"
            }
            score = (
                cosine_score_from_distance(r.get("_distance")) if query_vector else 0.0
            )
            for ep_id in memcell_to_eps.get(mc_id, ()):
                buckets[ep_id].append(
                    FactCandidate(
                        id=fid,
                        parent_episode_id=ep_id,
                        score=score,
                        metadata=metadata,
                    )
                )
        # Per-bucket cap; with query_vector the rows arrive sorted by
        # cosine ascending (closest first) so slicing keeps the most
        # relevant facts per episode.
        return {ep_id: bucket[:per_episode] for ep_id, bucket in buckets.items()}


def _q(value: str) -> str:
    return value.replace("'", "''")
