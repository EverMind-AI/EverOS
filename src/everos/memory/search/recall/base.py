"""Search hits -> everalgo ``Candidate``: the inbound mapper of the search path.

Every recaller turns the domain matches its index returns
(:class:`~everos.memory.ports.SearchHit`) into everalgo ``Candidate``
objects for the rank algorithms, and hands back the matched records next
to them in a :class:`RecallResult`. The two travel separately on purpose:

* ``Candidate.metadata`` carries only what an algorithm reads (passage
  text, parent links, timestamps) -- it is serialised verbatim into LLM
  rerank prompts, so it must not hold anything else;
* ``RecallResult.stored_by_id`` maps each ``Candidate.id`` to its stored
  record, which the shaper turns into the response DTO.

A ``Candidate.id`` is the record's global index key, except where a
recaller documents another key (the MaxSim parent fetches key episodes by
``entry_id``); ``stored_by_id`` is always keyed the same way as the
candidates beside it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from everalgo.types import Candidate, RankInput
from pydantic import BaseModel

from everos.component.tokenizer import Tokenizer
from everos.memory.ports import Filter, SearchHit, Stored

type CandidateSource = Literal["keyword", "vector", "other"]


@dataclasses.dataclass(frozen=True)
class RecallerDeps:
    """Shared dependencies for every index-backed recaller.

    Frozen so the orchestrator can build one instance and hand it to
    every recaller without worrying about state divergence.
    """

    tokenizer: Tokenizer


@dataclasses.dataclass(frozen=True)
class RecallResult[T]:
    """Candidates for the rank algorithms plus the records they stand for.

    Attributes:
        candidates: Ranked recall candidates.
        stored_by_id: The stored record behind each candidate, by
            ``Candidate.id``.
    """

    candidates: list[Candidate]
    stored_by_id: Mapping[str, Stored[T]]

    @classmethod
    def empty(cls) -> RecallResult[T]:
        return cls(candidates=[], stored_by_id={})

    def within_radius(self, radius: float | None) -> RecallResult[T]:
        """Drop the candidates scoring below ``radius``; ``None`` keeps all."""
        if radius is None:
            return self
        return dataclasses.replace(
            self, candidates=[c for c in self.candidates if c.score >= radius]
        )

    def head(self, n: int) -> RecallResult[T]:
        """The first ``n`` candidates."""
        return dataclasses.replace(self, candidates=self.candidates[:n])


@runtime_checkable
class KindRecaller(Protocol):
    """One business kind, BM25 + vector recall over its index."""

    kind: ClassVar[str]
    """``episode`` / ``atomic_fact`` / ``agent_case`` / ``agent_skill``."""

    everalgo_memory_type: ClassVar[str]
    """``episodic`` / ``case`` / ``skill`` -- passed to ``RankInput.memory_type``."""

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[Any]: ...

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[Any]: ...


def to_candidate(
    stored: Stored[BaseModel],
    *,
    score: float | None,
    source: CandidateSource,
    key: str | None = None,
    **extra: Any,
) -> Candidate:
    """Map one stored record to a ``Candidate``.

    ``metadata`` is the record's domain fields plus its ``entry_id`` and any
    ``extra`` fields an algorithm needs; ``key`` overrides the global id.
    """
    metadata = {
        **stored.item.model_dump(mode="python"),
        "entry_id": stored.ref.entry_id,
        **extra,
    }
    return Candidate(
        id=key or stored.ref.id,
        score=0.0 if score is None else float(score),
        source=source,
        metadata=metadata,
    )


def from_hits[T: BaseModel](
    hits: Iterable[SearchHit[T]], *, source: CandidateSource
) -> RecallResult[T]:
    """Candidates keyed by global id, scored by the index."""
    candidates: list[Candidate] = []
    stored_by_id: dict[str, Stored[T]] = {}
    for hit in hits:
        candidates.append(to_candidate(hit.stored, score=hit.score, source=source))
        stored_by_id[hit.stored.ref.id] = hit.stored
    return RecallResult(candidates=candidates, stored_by_id=stored_by_id)


def merge_stored[T](*results: RecallResult[T]) -> dict[str, Stored[T]]:
    """Every record behind any of ``results``, by candidate id."""
    merged: dict[str, Stored[T]] = {}
    for result in results:
        merged.update(result.stored_by_id)
    return merged


def merge_by_id_max[T](
    primary: RecallResult[T], extra: RecallResult[T]
) -> RecallResult[T]:
    """Union by id, keeping the higher score.

    One pass over both lists, primary first, so a duplicate id inside
    ``primary`` also keeps its higher score and a tie keeps the primary
    candidate.
    """
    by_id: dict[str, Candidate] = {}
    for c in (*primary.candidates, *extra.candidates):
        existing = by_id.get(c.id)
        if existing is None or c.score > existing.score:
            by_id[c.id] = c
    return RecallResult(
        candidates=list(by_id.values()),
        stored_by_id=merge_stored(primary, extra),
    )


def rank_input(
    query: str,
    *,
    memory_type: str,
    sparse: RecallResult[Any],
    dense: RecallResult[Any],
    top_k: int,
    radius: float | None,
) -> RankInput:
    """The everalgo ``arank`` input for one kind's two recall routes."""
    return RankInput(
        query=query,
        memory_type=memory_type,  # type: ignore[arg-type]  # ClassVar[str] of a Literal set
        sparse_candidates=sparse.candidates,
        dense_candidates=dense.candidates,
        top_k=top_k,
        radius=radius,
    )
