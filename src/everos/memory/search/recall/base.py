"""Search hits -> everalgo ``Candidate``: the inbound mapper of the search path.

Every recaller turns the domain matches its index returns
(:class:`~everos.memory.ports.SearchHit`) into everalgo ``Candidate``
objects for the multi-round loop, and hands back the matched records next
to them in a :class:`RecallResult`. The two travel separately on purpose:

* ``Candidate.metadata`` carries only what the loop reads (passage text,
  parent links, timestamps) -- it is rendered into the decider's evidence,
  so it must not hold anything else;
* ``RecallResult.stored_by_id`` maps each ``Candidate.id`` to its stored
  record, which the shaper turns into the response DTO.

A ``Candidate.id`` is the record's global index key; ``stored_by_id`` is
always keyed the same way as the candidates beside it.

:func:`stable_sparse_top` is the BM25 cut the episode and picture recallers
take: the same hits in the same order on every call, whatever order the
index returns tied hits in.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Final, Literal

from everalgo.types import Candidate
from pydantic import BaseModel

from everos.component.tokenizer import Tokenizer
from everos.memory.ports import SearchHit, Stored

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
    """Candidates for the multi-round loop plus the records they stand for.

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

    def head(self, n: int) -> RecallResult[T]:
        """The first ``n`` candidates."""
        return dataclasses.replace(self, candidates=self.candidates[:n])


def to_candidate(
    stored: Stored[BaseModel],
    *,
    score: float | None,
    source: CandidateSource,
) -> Candidate:
    """Map one stored record to a ``Candidate`` keyed by its global id.

    ``metadata`` is the record's domain fields plus its ``entry_id``.
    """
    metadata = {
        **stored.item.model_dump(mode="python"),
        "entry_id": stored.ref.entry_id,
    }
    return Candidate(
        id=stored.ref.id,
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


BM25_TIE_DECIMALS: Final = 5
"""Two BM25 scores agreeing to this many decimals are a tie: LanceDB reports the
same row's score with ~1e-6 of noise from call to call."""

BM25_MAX_FETCH: Final = 32
""":func:`stable_sparse_top` widens its fetch to at most this many times ``limit``
while looking for the end of the tie group at the cut."""


def _bm25_order(hit: SearchHit[Any]) -> tuple[float, str]:
    """Score descending to :data:`BM25_TIE_DECIMALS`, then global id ascending."""
    return (-round(hit.score or 0.0, BM25_TIE_DECIMALS), hit.stored.ref.id)


async def stable_sparse_top[T](
    search: Callable[[int], Awaitable[Sequence[SearchHit[T]]]], limit: int
) -> list[SearchHit[T]]:
    """The top ``limit`` BM25 hits, as the same hits in the same order on every call.

    BM25 ties are the rule on short units: two utterances matching the same terms
    at the same length score identically, LanceDB returns tied rows in no fixed
    order, and it reports a row's score with ~1e-6 of noise from call to call. A
    plain cut at ``limit`` therefore keeps a different subset of the tie group at
    the boundary, and the RRF order downstream moves with it -- measured on a
    Time-Dialog store (2026-10-23): 47% of questions saw a different round-0
    candidate list across 4 identical concurrent searches.

    So the cut is taken here: fetch past ``limit`` until the tie group at the
    boundary is complete (the last fetched score is strictly lower), order by
    (score rounded to :data:`BM25_TIE_DECIMALS`, id) and cut. Widening gives up
    at :data:`BM25_MAX_FETCH` times ``limit`` and cuts what it has. This is the
    exp tree's ``fts_rows`` rule, applied through the index port instead of on a
    LanceDB table: the parity check of 2026-11-10 (deviation R3) found this tree
    taking the engine's order, so the same search returned a different list
    from call to call where the exp tree returned one. On a multi-column index
    (a picture's description and text) the port merges the columns by best
    score first; cutting the merged list here keeps the same hits, in the same
    order, as cutting each column and then the merge, which is how the exp tree's
    2.0 picture port does it.

    Args:
        search: Runs the index's BM25 search with the given row limit.
        limit: How many hits to keep.
    """
    if limit <= 0:
        return []
    cap = limit * BM25_MAX_FETCH
    fetch = min(max(2 * limit, limit + 32), cap)
    while True:
        hits = sorted(await search(fetch), key=_bm25_order)
        if len(hits) < fetch or fetch >= cap:
            return hits[:limit]
        if _bm25_order(hits[-1])[0] > _bm25_order(hits[limit - 1])[0]:
            return hits[:limit]  # the tie group at the cut is complete
        fetch = min(fetch * 2, cap)
