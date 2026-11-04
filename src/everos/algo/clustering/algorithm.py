"""Cluster operators — geometry and LLM-refined paths over a list[Cluster] state.

``cluster_by_geometry`` is sync pure-compute (cosine + time-window, no I/O). ``acluster_by_llm`` is native
async (it calls an LLM). Both are stateless: accept a
new_cluster + existing_clusters list, return a merged Cluster or None without mutating any input. The caller
owns persistence and concurrency control.
"""

from __future__ import annotations

__all__ = ["acluster_by_llm", "cluster_by_geometry"]

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from everos.algo.clustering._replies import _call_llm_for_cluster_assignment
from everos.algo.clustering.prompts.en.cluster import (
    CLUSTER_LLM_ASSIGN_USER_PROMPT,
    assemble_system_prompt,
)
from everos.algo.clustering.state import Cluster
from everos.algo.llm import LLMMessage
from everos.algo.utils import render_prompt

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)

_MS_PER_DAY = 86_400_000
_NORM_EPSILON = 1e-9


def cluster_by_geometry(
    new_cluster: Cluster,
    existing_clusters: list[Cluster],
    *,
    threshold: float = 0.65,
    time_window_days: float = 7.0,
    preview_cap: int = 5,
) -> Cluster | None:
    """Cosine + time-window assignment.

    Args:
        new_cluster: Size-1 Cluster representing the incoming item.
        existing_clusters: Caller-owned list; never mutated.
        threshold: Cosine similarity >= this assigns to top-1. Default 0.65.
        time_window_days: Clusters whose last_ts is older than this gap are excluded. Default 7.0.
        preview_cap: Maximum preview length on the merged snapshot. Default 5.

    Returns:
        Merged Cluster when assigned to an existing one (id is passed through from
        existing_clusters[i].id), or None when no match — caller appends new_cluster as a brand-new
        cluster and mints its own id.
    """
    if not existing_clusters:
        return None

    idx = _find_best_within_window(
        existing_clusters,
        new_cluster.centroid,
        new_cluster.last_ts,
        threshold,
        time_window_days,
    )
    if idx is None:
        return None
    return _merge(existing_clusters[idx], new_cluster, preview_cap=preview_cap)


async def acluster_by_llm(
    new_cluster: Cluster,
    existing_clusters: list[Cluster],
    *,
    llm: LLMClient,
    k_candidates: int = 30,
    llm_skip_threshold: float = 0.85,
    custom_instructions: str | None = None,
    preview_cap: int = 5,
) -> Cluster | None:
    """LLM-refined assignment: top-K recall → fast path → LLM ranking by idx.

    LLM ranking query text comes from new_cluster.preview[0] if preview else "".

    Args:
        new_cluster: Size-1 Cluster representing the incoming item.
        existing_clusters: Caller-owned list; never mutated.
        llm: LLM client for the Stage-2 ranking call.
        k_candidates: Number of nearest clusters fetched before LLM ranking. Default 30.
        llm_skip_threshold: Skip LLM and assign to top-1 when its similarity meets this. Default 0.85.
        custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
            message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are never
            rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system message
            unchanged. The user message always carries the data.
        preview_cap: Maximum preview length on the merged snapshot. Default 5.

    Returns:
        Merged Cluster when LLM ranks a match (id passed through), or None when LLM picks
        "new" — caller appends new_cluster as a brand-new cluster and mints its own id.
        LLM internally uses idx; idx -> existing_clusters[idx].id transparent to caller.

    Raises:
        LLMOutputError: If the LLM reply is not a valid ``{"idx", "reason"}`` object.
    """
    if not existing_clusters:
        return None

    candidates = _find_top_k_clusters(
        existing_clusters, new_cluster.centroid, k_candidates
    )
    if not candidates:
        return None

    top_idx, top_sim = candidates[0]

    if top_sim >= llm_skip_threshold:
        return _merge(existing_clusters[top_idx], new_cluster, preview_cap=preview_cap)

    query_text = new_cluster.preview[0] if new_cluster.preview else ""
    candidate_indices = [idx for idx, _ in candidates]
    clusters_json = _build_clusters_json(existing_clusters, candidate_indices)
    messages = _assignment_request(query_text, clusters_json, custom_instructions)
    chosen_idx = (await _call_llm_for_cluster_assignment(llm, messages)).idx
    if 0 <= chosen_idx < len(existing_clusters):
        return _merge(
            existing_clusters[chosen_idx], new_cluster, preview_cap=preview_cap
        )
    return None


# --- private helpers --------------------------------------------------------


def _assignment_request(
    query_text: str, clusters_json: str, custom_instructions: str | None
) -> list[LLMMessage]:
    """The assignment call's ``[system, user]`` pair: the rules, then the candidate clusters and the new intent."""
    user = render_prompt(
        CLUSTER_LLM_ASSIGN_USER_PROMPT,
        None,
        clusters_json=clusters_json,
        memcell_text=query_text,
    )
    return [
        LLMMessage(
            role="system",
            content=assemble_system_prompt(custom_instructions=custom_instructions),
        ),
        LLMMessage(role="user", content=user),
    ]


def _merge(existing: Cluster, new: Cluster, *, preview_cap: int) -> Cluster:
    """Weighted centroid merge + preview concat + members append."""
    total = existing.count + new.count
    return Cluster(
        id=existing.id,  # transparent passthrough — algo never generates id
        centroid=(existing.centroid * existing.count + new.centroid * new.count)
        / total,
        count=total,
        last_ts=max(existing.last_ts, new.last_ts),
        preview=(existing.preview + new.preview)[-preview_cap:],
        members=existing.members + new.members,
    )


def _cosine(v: np.ndarray, c: np.ndarray) -> float:
    """Cosine similarity with _NORM_EPSILON divide-by-zero guard."""
    v_norm = float(np.linalg.norm(v)) + _NORM_EPSILON
    c_norm = float(np.linalg.norm(c)) + _NORM_EPSILON
    return float(np.dot(v, c) / (v_norm * c_norm))


def _find_best_within_window(
    clusters: list[Cluster],
    vector: np.ndarray,
    timestamp_ms: int,
    threshold: float,
    time_window_days: float,
) -> int | None:
    """Top-1 cluster index within the time window clearing threshold; None otherwise."""
    max_gap_ms = int(time_window_days * _MS_PER_DAY)
    best_idx: int | None = None
    best_sim = -1.0

    for i, cluster in enumerate(clusters):
        if cluster.centroid.size == 0:
            continue
        if abs(timestamp_ms - cluster.last_ts) > max_gap_ms:
            continue
        sim = _cosine(vector, cluster.centroid)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_idx is not None and best_sim >= threshold:
        return best_idx
    return None


def _find_top_k_clusters(
    clusters: list[Cluster],
    vector: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Top-K clusters by cosine similarity (no time-window filter); empty-centroid clusters skipped."""
    scored: list[tuple[int, float]] = []
    for i, cluster in enumerate(clusters):
        if cluster.centroid.size == 0:
            continue
        scored.append((i, _cosine(vector, cluster.centroid)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _build_clusters_json(clusters: list[Cluster], candidate_indices: list[int]) -> str:
    """Serialise candidate clusters to JSON for the ``{{clusters_json}}`` user-prompt placeholder."""
    if not candidate_indices:
        return "(No existing clusters)"

    items = [
        {
            "idx": idx,
            "count": clusters[idx].count,
            "preview": clusters[idx].preview,
        }
        for idx in candidate_indices
    ]
    return json.dumps(items, ensure_ascii=False, indent=2)
