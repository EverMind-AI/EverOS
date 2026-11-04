"""MaxSim retrieval — child-first, group-by-parent, max-pool facade.

Caller injects ``child_retrieve`` (any ``RetrieveFn`` — typically vector or BM25 over child
chunks) and ``parent_fetch`` (a batch loader keyed by parent ID); this operator runs child
recall, groups results by ``metadata["parent_id"]``, max-pools per parent, and returns fully
hydrated parent candidates. Pure algorithm orchestration: no LLM, no index I/O beyond the
two caller-supplied callables.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everos.algo.rank.protocols import RetrieveFn
    from everos.algo.types import Candidate

__all__ = ["ParentFetchFn", "amaxsim_retrieve"]

ParentFetchFn = Callable[[list[str]], Awaitable[list["Candidate"]]]
"""``fetch(parent_ids) -> list[Candidate]`` — batch loader for parent documents by ID."""


async def amaxsim_retrieve(
    query: str,
    *,
    child_retrieve: RetrieveFn,
    parent_fetch: ParentFetchFn,
    top_n: int = 20,
    child_candidates: int = 200,
    min_score: float | None = None,
) -> list[Candidate]:
    """Child-first MaxSim aggregation retrieval — group children by parent_id, max-pool, fetch parents.

    Runs child-level recall first (finer-grained chunks carry better query signal), aggregates via
    max-pool so each parent inherits its strongest child score, then batch-fetches and rescores
    parent documents before returning the ranked list.

    Args:
        query: The search query string passed verbatim to ``child_retrieve``.
        child_retrieve: Async callable ``(query, k) -> list[Candidate]`` for child-level recall.
            Each returned ``Candidate`` must carry ``metadata["parent_id"]`` (a ``str``) to be
            included in aggregation; children without a valid string ``parent_id`` are silently
            skipped.
        parent_fetch: Async callable ``(parent_ids) -> list[Candidate]`` — batch-loads parent
            documents by ID. The returned list may be a subset of the requested IDs (missing
            parents are naturally absent from the result). ``source`` and ``metadata`` on the
            returned ``Candidate`` objects are preserved verbatim.
        top_n: Maximum number of parent candidates to fetch and return (default 20). Applied
            after max-pool sort, before ``parent_fetch``.
        child_candidates: How many children to request from ``child_retrieve`` (default 200).
            A larger pool increases parent recall at the cost of one larger recall call.
        min_score: When set, parents whose max-pool score is strictly below this threshold are
            filtered from the final result before returning (default ``None`` — no filtering).

    Returns:
        Parent candidates sorted descending by their max-pool child score, at most ``top_n``
        entries. Each ``Candidate`` carries the original ``source`` and ``metadata`` from
        ``parent_fetch``, with ``score`` replaced by the max-pool value.

    Algorithm:
        1. Call ``child_retrieve(query, child_candidates)``; return ``[]`` if empty.
        2. Group children by ``metadata["parent_id"]``; skip any child missing a string parent_id.
           For each parent, keep the maximum child score (``float("-inf")`` sentinel, strict ``>``).
        3. Return ``[]`` if no valid parent groups were found.
        4. Sort parents by max score descending (stable Timsort) and slice ``[:top_n]``.
        5. Call ``parent_fetch(top_ids)`` to hydrate the parent documents.
        6. Rescore each returned parent with its max-pool score from step 4.
        7. Apply ``min_score`` filter if set (``score >= min_score``).
        8. Sort rescored list descending by score and return.

    Resource contract:
        ``child_retrieve`` is called exactly once with ``child_candidates``. ``parent_fetch``
        is called exactly once with the top-N parent IDs. No LLM calls are made. No filesystem
        or database I/O beyond the two caller-supplied callables.
    """
    from everos.algo.types import Candidate

    children = await child_retrieve(query, child_candidates)
    if not children:
        return []

    parent_max: dict[str, float] = {}
    for c in children:
        pid = c.metadata.get("parent_id")
        if not isinstance(pid, str):
            continue
        if c.score > parent_max.get(pid, float("-inf")):
            parent_max[pid] = c.score

    if not parent_max:
        return []

    ranked = sorted(parent_max.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    top_ids = [pid for pid, _ in ranked]
    score_by_id = dict(ranked)

    parents = await parent_fetch(top_ids)

    rescored = [
        Candidate(
            id=p.id,
            score=score_by_id.get(p.id, 0.0),
            source=p.source,
            metadata=p.metadata,
        )
        for p in parents
    ]

    if min_score is not None:
        rescored = [c for c in rescored if c.score >= min_score]

    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored
