"""Cluster Manager - Pure computation component for memcell clustering.

This module provides ClusterManager, a pure computation component that clusters
memcells based on semantic similarity and temporal proximity.

IMPORTANT: This is a pure computation component. The caller is responsible
for loading/saving mem scene state.

Usage:
    from memory_layer.cluster_manager import ClusterManager, ClusterManagerConfig, MemSceneState

    # Initialize
    config = ClusterManagerConfig(
        similarity_threshold=0.65,
        max_time_gap_days=7,
    )
    cluster_mgr = ClusterManager(config)

    # Caller loads state (from InMemory / MongoDB / file)
    state_dict = await storage.load_mem_scene(group_id)
    state = MemSceneState.from_dict(state_dict) if state_dict else MemSceneState()

    # Pure computation
    cluster_id, state = await cluster_mgr.cluster_memcell(memcell, state)

    # Caller saves state
    await storage.save_mem_scene(group_id, state.to_dict())
"""

from memory_layer.cluster_manager.config import ClusterManagerConfig
from memory_layer.cluster_manager.manager import ClusterManager, MemSceneState

__all__ = ["ClusterManager", "ClusterManagerConfig", "MemSceneState"]
