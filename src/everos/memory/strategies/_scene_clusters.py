"""Where an episode goes among its owner's scene clusters.

Shared by the two scene strategies -- a user's (``user_scene``) and a
group's (``group_scene``): each new episode is folded into its owner's scene
clusters with everalgo's ``cluster_by_geometry`` (``[clustering]`` threshold
and time window), the geometry users' episode clusters use, or starts a
cluster of its own. A cluster of enough episodes is a scene; the narrative
is each strategy's own call.
"""

from __future__ import annotations

__all__ = ["place_in_scenes"]

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_geometry

from everos.memory.deps import MemoryDeps
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import ClusterKind, mint_cluster_id
from everos.memory.ports import Scope


async def place_in_scenes(
    deps: MemoryDeps,
    scope: Scope,
    kind: ClusterKind,
    *,
    entry_id: str,
    text: str,
    timestamp_ms: int,
    vector: np.ndarray,
    threshold: float,
    time_window_days: float,
) -> AlgoCluster:
    """The scene cluster that holds the episode; placed and persisted if new.

    A replay (or a retry after the narrative call failed) finds the episode
    placed already and gets that cluster back unchanged, so a member is never
    folded in twice. The caller holds the owner's scene lock.
    """
    existing = [
        cluster_mapper.to_algo(c)
        for c in await deps.stores.cluster.list_for_owner(scope, kind)
    ]
    placed = next((c for c in existing if entry_id in c.members), None)
    if placed is not None:
        return placed
    single = AlgoCluster(
        id=mint_cluster_id(),
        centroid=vector,
        count=1,
        last_ts=timestamp_ms,
        preview=[text],
        members=[entry_id],
    )
    cluster = (
        cluster_by_geometry(
            single, existing, threshold=threshold, time_window_days=time_window_days
        )
        or single
    )
    await deps.stores.cluster.upsert(
        scope, kind, cluster_mapper.from_algo(cluster), member_type="episode"
    )
    return cluster
