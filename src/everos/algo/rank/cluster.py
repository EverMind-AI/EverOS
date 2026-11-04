"""Cluster-scoped retriever composer for ``aagentic_retrieve``.

``acluster_retrieve`` scans the base retrieval results in score order and accumulates
distinct clusters until ``cluster_top_k`` are reached (first-hit strategy), then
expands those clusters' full memberships from a caller-provided ``all_docs`` list and
returns the expansion unreranked. Rerank is intentionally left to the caller
(typically ``aagentic_retrieve`` rerunning the same ``rerank_fn`` over Round 1 and
again over the merged Round 1 + Round 2 set), which avoids the double-rerank trap and
keeps the Round 2 final rerank effective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.clustering import Cluster
    from everos.algo.rank.protocols import RetrieveFn
    from everos.algo.types.rank import Candidate

__all__ = ["acluster_retrieve"]


async def acluster_retrieve(
    query: str,
    *,
    base_retrieve: RetrieveFn,
    base_candidates: int | None = None,
    clusters: Sequence[Cluster],
    all_docs: Sequence[Candidate],
    cluster_top_k: int = 5,
) -> list[Candidate]:
    """Narrow full-corpus retrieve to top-K cluster scope and return the raw expansion.

    Algorithm:
        1. ``base_retrieve(query, base_candidates)`` — wide candidate pool for cluster
           selection (Level 1). Caller typically passes a hybrid (dense+sparse RRF)
           closure.
        2. Scan results in score order, accumulating distinct clusters (first-hit): the
           first member of a cluster seen determines that cluster's inclusion. Stop as
           soon as ``cluster_top_k`` distinct clusters are reached.
        3. Expand the selected clusters' full memberships from ``all_docs`` (id lookup)
           and return in ``all_docs`` order (no rerank, no truncation).

    Args:
        query: Search query.
        base_retrieve: Async ``(query, k) -> list[Candidate]`` used to seed cluster
            selection. Caller typically passes a hybrid (dense+sparse RRF) closure.
        base_candidates: Top-k passed to ``base_retrieve``. ``None`` means no
            truncation (passes ``len(all_docs)`` — the corpus upper bound). When set,
            typically 50-100; must be wide enough to cover ≥ ``cluster_top_k`` distinct
            clusters.
        clusters: All clusters in the corpus. Each cluster contributes its members at
            most once; the algorithm reads ``c.id`` and ``c.members``.
        all_docs: Full corpus snapshot as ``Candidate`` list. Used to expand selected
            clusters' members — items not in any selected cluster are dropped. Caller
            owns this list (small per-conversation corpus; OK to pass as plain list).
        cluster_top_k: How many top-scoring clusters to keep.

    Returns:
        Expanded members of the top-``cluster_top_k`` clusters, in ``all_docs`` order,
        unranked and unscored (each candidate's ``score`` is whatever ``all_docs``
        carried — caller should not rely on it). Empty list when no base hit maps to
        any cluster.

    Resource contract:
        - ``base_retrieve`` is called exactly once.
        - No rerank calls.
        - No LLM calls.
    """
    base_results = await base_retrieve(
        query, base_candidates if base_candidates is not None else len(all_docs)
    )
    if not base_results:
        return []

    # Reverse map: member_id -> cluster_id (built per call; clusters list is tiny).
    member_to_cluster: dict[str, str] = {
        member: cluster.id
        for cluster in clusters
        if cluster.id is not None
        for member in cluster.members
    }

    selected_cluster_ids: set[str] = set()
    for cand in base_results:
        cluster_id = member_to_cluster.get(cand.id)
        if cluster_id is None:
            continue
        selected_cluster_ids.add(cluster_id)
        if len(selected_cluster_ids) >= cluster_top_k:
            break

    if not selected_cluster_ids:
        return []

    # Expand selected clusters' full memberships from all_docs (original docs order).
    member_ids: set[str] = set()
    for cluster in clusters:
        if cluster.id in selected_cluster_ids:
            member_ids.update(cluster.members)

    return [d for d in all_docs if d.id in member_ids]
