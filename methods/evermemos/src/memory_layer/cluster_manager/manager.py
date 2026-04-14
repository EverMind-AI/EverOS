"""ClusterManager - Core component for automatic memcell clustering.

This module provides pure computation logic for clustering memcells.
Storage is managed by the caller, not by ClusterManager itself.

Design:
- ClusterManager is a pure computation component
- Input: memcell + current state
- Output: cluster_id + updated state
- Caller is responsible for loading/saving state
"""

import asyncio
import json
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from memory_layer.cluster_manager.config import ClusterManagerConfig
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Try to import vectorize service
try:
    from agentic_layer.vectorize_service import get_vectorize_service

    VECTORIZE_SERVICE_AVAILABLE = True
except ImportError:
    VECTORIZE_SERVICE_AVAILABLE = False
    logger.warning("Vectorize service not available, clustering will be limited")



class MemSceneState:
    """Internal state for a single group's clustering."""

    def __init__(self):
        """Initialize empty mem scene state."""
        self.event_ids: List[str] = []
        self.timestamps: List[float] = []
        self.vectors: List[np.ndarray] = []
        self.cluster_ids: List[str] = []
        self.eventid_to_cluster: Dict[str, str] = {}
        self.next_cluster_idx: int = 0

        # Centroid-based clustering state
        self.cluster_centroids: Dict[str, np.ndarray] = {}
        self.cluster_counts: Dict[str, int] = {}
        self.cluster_last_ts: Dict[str, Optional[float]] = {}

        # Clusters that contain agent conversation memcells
        self.case_cluster_ids: set = set()

    def assign_new_cluster(self, event_id: str) -> str:
        """Assign a new cluster ID to an event."""
        cluster_id = f"cluster_{self.next_cluster_idx:03d}"
        self.next_cluster_idx += 1
        self.eventid_to_cluster[event_id] = cluster_id
        self.cluster_ids.append(cluster_id)
        return cluster_id

    def add_to_cluster(
        self,
        event_id: str,
        cluster_id: str,
        vector: np.ndarray,
        timestamp: Optional[float],
    ) -> None:
        """Add an event to an existing cluster."""
        self.eventid_to_cluster[event_id] = cluster_id
        self.cluster_ids.append(cluster_id)
        self._update_cluster_centroid(cluster_id, vector, timestamp)

    def _update_cluster_centroid(
        self, cluster_id: str, vector: np.ndarray, timestamp: Optional[float]
    ) -> None:
        """Update cluster centroid with new vector."""
        if vector is None or vector.size == 0:
            if timestamp is not None:
                prev_ts = self.cluster_last_ts.get(cluster_id)
                self.cluster_last_ts[cluster_id] = max(prev_ts or timestamp, timestamp)
            return

        count = self.cluster_counts.get(cluster_id, 0)
        if count <= 0:
            self.cluster_centroids[cluster_id] = vector.astype(np.float32, copy=False)
            self.cluster_counts[cluster_id] = 1
        else:
            current_centroid = self.cluster_centroids[cluster_id]
            if current_centroid.dtype != np.float32:
                current_centroid = current_centroid.astype(np.float32)
            new_centroid = (current_centroid * float(count) + vector) / float(count + 1)
            self.cluster_centroids[cluster_id] = new_centroid.astype(
                np.float32, copy=False
            )
            self.cluster_counts[cluster_id] = count + 1

        if timestamp is not None:
            prev_ts = self.cluster_last_ts.get(cluster_id)
            self.cluster_last_ts[cluster_id] = max(prev_ts or timestamp, timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Produces the new format with memcell_info and memscene_info maps.
        """
        memcell_info = {}
        for i, event_id in enumerate(self.event_ids):
            memcell_info[event_id] = {
                "memscene": self.eventid_to_cluster.get(event_id, ""),
                "timestamp": self.timestamps[i] if i < len(self.timestamps) else 0.0,
            }

        all_cids = (
            set(self.cluster_centroids.keys())
            | set(self.cluster_counts.keys())
            | set(self.cluster_last_ts.keys())
        )
        memscene_info = {}
        for cid in all_cids:
            memscene_info[cid] = {
                "center": (
                    self.cluster_centroids[cid].tolist()
                    if cid in self.cluster_centroids
                    else []
                ),
                "timestamp": self.cluster_last_ts.get(cid),
                "count": self.cluster_counts.get(cid, 0),
            }

        result = {
            "memcell_info": memcell_info,
            "memscene_info": memscene_info,
            "next_cluster_idx": self.next_cluster_idx,
        }
        if self.case_cluster_ids:
            result["case_cluster_ids"] = sorted(self.case_cluster_ids)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemSceneState":
        """Create MemSceneState from dictionary.

        Supports both new format (memcell_info/memscene_info) and old format
        (event_ids/timestamps/cluster_ids/...) for backward compatibility.
        """
        state = MemSceneState()
        state.next_cluster_idx = int(data.get("next_cluster_idx", 0))
        state.case_cluster_ids = set(data.get("case_cluster_ids") or [])

        if "memcell_info" in data:
            # New format
            for event_id, info in data["memcell_info"].items():
                state.event_ids.append(event_id)
                state.timestamps.append(float(info.get("timestamp", 0.0)))
                cluster_id = info.get("memscene", "")
                state.cluster_ids.append(cluster_id)
                state.eventid_to_cluster[event_id] = cluster_id

            for cid, info in data.get("memscene_info", {}).items():
                center = info.get("center", [])
                if center:
                    state.cluster_centroids[cid] = np.array(center, dtype=np.float32)
                ts = info.get("timestamp")
                state.cluster_last_ts[cid] = float(ts) if ts is not None else None
                state.cluster_counts[cid] = int(info.get("count", 0))
        else:
            # Old format (backward compatibility)
            state.event_ids = list(data.get("event_ids", []))
            state.timestamps = list(data.get("timestamps", []))
            state.cluster_ids = list(data.get("cluster_ids", []))
            state.eventid_to_cluster = dict(data.get("eventid_to_cluster", {}))

            centroids = data.get("cluster_centroids", {}) or {}
            state.cluster_centroids = {
                k: np.array(v, dtype=np.float32) for k, v in centroids.items()
            }
            state.cluster_counts = {
                k: int(v) for k, v in (data.get("cluster_counts", {}) or {}).items()
            }
            state.cluster_last_ts = {
                k: float(v) for k, v in (data.get("cluster_last_ts", {}) or {}).items()
            }

        return state


class ClusterManager:
    """Automatic clustering manager - pure computation component.

    ClusterManager handles incremental clustering of memcells based on semantic
    similarity (embeddings) and temporal proximity.

    IMPORTANT: This is a pure computation component. The caller is responsible
    for loading/saving mem scene state.

    Usage:
        ```python
        cluster_mgr = ClusterManager(config)

        # Caller loads state (from InMemory / MongoDB / file)
        state_dict = await storage.load(group_id)
        state = MemSceneState.from_dict(state_dict) if state_dict else MemSceneState()

        # Pure computation
        cluster_id, updated_state = await cluster_mgr.cluster_memcell(memcell, state)

        # Caller saves state
        await storage.save(group_id, updated_state.to_dict())
        ```
    """

    def __init__(
        self,
        config: Optional[ClusterManagerConfig] = None,
        llm_provider: Optional[Any] = None,
        context_fetcher: Optional[Callable] = None,
    ):
        """Initialize ClusterManager.

        Args:
            config: Clustering configuration (uses defaults if None)
            llm_provider: LLM provider instance (required for agent memcell clustering)
            context_fetcher: Async callback to fetch context texts from DB.
                Signature: (event_ids: List[str]) -> Dict[str, str]
                Returns mapping of event_id -> task_intent text.
                Required for agent memcell clustering.
        """
        self.config = config or ClusterManagerConfig()
        self._callbacks: List[Callable] = []

        # Vectorize service (for embedding)
        self._vectorize_service = None
        if VECTORIZE_SERVICE_AVAILABLE:
            try:
                self._vectorize_service = get_vectorize_service()
            except Exception as e:
                logger.warning(f"Failed to initialize vectorize service: {e}")

        # LLM provider (for llm algorithm)
        self._llm_provider = llm_provider
        self._context_fetcher = context_fetcher

        # Statistics
        self._stats = {
            "total_memcells": 0,
            "clustered_memcells": 0,
            "new_clusters": 0,
            "failed_embeddings": 0,
        }

    def on_cluster_assigned(
        self, callback: Callable[[str, Dict[str, Any], str], None]
    ) -> None:
        """Register a callback for cluster assignment events.

        Callback signature:
            callback(group_id: str, memcell: Dict[str, Any], cluster_id: str) -> None
        """
        self._callbacks.append(callback)

    async def cluster_memcell(
        self,
        memcell: Dict[str, Any],
        state: MemSceneState,
        has_case: bool = False,
    ) -> Tuple[Optional[str], MemSceneState]:
        """Cluster a memcell and return updated state.

        Caller is responsible for loading state before and saving it after.

        Routing:
        - has_case=False: embedding clustering over non-case clusters, text=episode
        - has_case=True:  embedding recall + LLM over case clusters, text=task_intent

        Args:
            memcell: Memcell dictionary with event_id, timestamp, episode/summary
            state: Current mem scene state for the group
            has_case: Whether this memcell has an agent case

        Returns:
            Tuple of (cluster_id, updated_state):
            - cluster_id: Assigned cluster ID, or None if failed
            - state: Updated MemSceneState (same object, mutated)
        """
        if has_case:
            return await self._cluster_memcell_llm(memcell, state)
        return await self._cluster_memcell_embedding(memcell, state)

    async def _cluster_memcell_embedding(
        self, memcell: Dict[str, Any], state: MemSceneState
    ) -> Tuple[Optional[str], MemSceneState]:
        """Embedding-based clustering using vector cosine similarity."""
        self._stats["total_memcells"] += 1

        # Extract key fields
        event_id = str(memcell.get("event_id", ""))
        if not event_id:
            logger.warning("Memcell missing event_id, skipping clustering")
            return None, state

        timestamp = self._parse_timestamp(memcell.get("timestamp"))
        text = self._extract_text(memcell)

        # Get embedding
        vector = await self._get_embedding(text)
        if vector is None or vector.size == 0:
            logger.warning(
                f"Failed to get embedding for event {event_id}, creating singleton cluster"
            )
            cluster_id = state.assign_new_cluster(event_id)
            state.event_ids.append(event_id)
            state.timestamps.append(timestamp or 0.0)
            state.vectors.append(np.zeros((1,), dtype=np.float32))
            self._stats["new_clusters"] += 1
            self._stats["failed_embeddings"] += 1
            return cluster_id, state

        # Find best matching cluster (exclude case clusters)
        cluster_id = self._find_best_cluster(
            state, vector, timestamp, exclude_cids=state.case_cluster_ids
        )

        # Add to cluster
        if cluster_id is None:
            cluster_id = state.assign_new_cluster(event_id)
            state._update_cluster_centroid(cluster_id, vector, timestamp)
            self._stats["new_clusters"] += 1
        else:
            state.add_to_cluster(event_id, cluster_id, vector, timestamp)

        # Update state
        state.event_ids.append(event_id)
        state.timestamps.append(timestamp or 0.0)
        state.vectors.append(vector)

        self._stats["clustered_memcells"] += 1

        return cluster_id, state

    def _create_new_cluster(
        self,
        state: MemSceneState,
        event_id: str,
        vector: Optional[np.ndarray],
        timestamp: Optional[float],
        is_case: bool = False,
    ) -> str:
        """Create a new cluster and assign the event to it."""
        cluster_id = state.assign_new_cluster(event_id)
        if is_case:
            state.case_cluster_ids.add(cluster_id)
        # _update_cluster_centroid handles cluster_counts when vector is present;
        # for None/empty vector we must set it explicitly.
        if vector is not None and vector.size > 0:
            state._update_cluster_centroid(cluster_id, vector, timestamp)
        else:
            state.cluster_counts[cluster_id] = 1
            if timestamp is not None:
                state.cluster_last_ts[cluster_id] = timestamp
        self._stats["new_clusters"] += 1
        return cluster_id

    def _assign_to_cluster(
        self,
        state: MemSceneState,
        event_id: str,
        cluster_id: str,
        vector: Optional[np.ndarray],
        timestamp: Optional[float],
    ) -> None:
        """Assign an event to an existing cluster."""
        state.eventid_to_cluster[event_id] = cluster_id
        state.cluster_ids.append(cluster_id)
        # _update_cluster_centroid handles cluster_counts when vector is present;
        # for None/empty vector we must increment explicitly.
        if vector is not None and vector.size > 0:
            state._update_cluster_centroid(cluster_id, vector, timestamp)
        else:
            state.cluster_counts[cluster_id] = (
                state.cluster_counts.get(cluster_id, 0) + 1
            )
            if timestamp is not None:
                prev_ts = state.cluster_last_ts.get(cluster_id)
                state.cluster_last_ts[cluster_id] = max(
                    prev_ts or timestamp, timestamp
                )

    def _append_event(
        self,
        state: MemSceneState,
        event_id: str,
        vector: Optional[np.ndarray],
        timestamp: Optional[float],
    ) -> None:
        """Append event metadata to state lists."""
        state.event_ids.append(event_id)
        state.timestamps.append(timestamp or 0.0)
        state.vectors.append(
            vector if vector is not None else np.zeros((1,), dtype=np.float32)
        )

    async def _cluster_memcell_llm(
        self,
        memcell: Dict[str, Any],
        state: MemSceneState,
    ) -> Tuple[Optional[str], MemSceneState]:
        """LLM-based clustering with embedding pre-filtering.

        Two-stage approach:
        1. Use embedding similarity to recall top-K candidate clusters
        2. Fetch recent episodes for candidates, let LLM make the final decision
        """
        self._stats["total_memcells"] += 1

        event_id = str(memcell.get("event_id", ""))
        if not event_id:
            logger.warning("Memcell missing event_id, skipping clustering")
            return None, state

        timestamp = self._parse_timestamp(memcell.get("timestamp"))
        text = self._extract_text(memcell)

        if self._llm_provider is None:
            logger.error(
                "[LLM Clustering] No LLM provider configured, "
                "falling back to embedding-only case clustering"
            )
            vector = await self._get_embedding(text)
            best_cid = self._find_top_k_clusters(
                state, vector, k=1, only_cids=state.case_cluster_ids,
            )
            if best_cid and best_cid[0][1] >= self.config.similarity_threshold:
                cluster_id = best_cid[0][0]
                self._assign_to_cluster(
                    state, event_id, cluster_id, vector, timestamp
                )
            else:
                cluster_id = self._create_new_cluster(
                    state, event_id, vector, timestamp, is_case=True
                )
            self._append_event(state, event_id, vector, timestamp)
            self._stats["clustered_memcells"] += 1
            return cluster_id, state

        # No existing case clusters — just create a new one directly
        if not state.case_cluster_ids:
            vector = await self._get_embedding(text)
            cluster_id = self._create_new_cluster(
                state, event_id, vector, timestamp, is_case=True
            )
            self._append_event(state, event_id, vector, timestamp)
            self._stats["clustered_memcells"] += 1
            logger.info(
                f"[LLM Clustering] First case cluster: {event_id} -> {cluster_id}"
            )
            return cluster_id, state

        # Stage 1: Embedding recall — find top-K candidate clusters (case only)
        vector = await self._get_embedding(text)
        scored_candidates = self._find_top_k_clusters(
            state, vector,
            k=self.config.llm_top_k_clusters,
            only_cids=state.case_cluster_ids,
        )
        candidate_ids = [cid for cid, _ in scored_candidates]
        top1_sim = scored_candidates[0][1] if scored_candidates else -1.0
        logger.info(
            f"[LLM Clustering] Embedding recall: {len(candidate_ids)} candidates "
            f"(top1_sim={top1_sim:.3f}), "
            f"from {len(state.case_cluster_ids)} case clusters"
        )

        # Fast path: if top-1 similarity is high enough, skip LLM
        if top1_sim >= self.config.llm_skip_threshold:
            cluster_id = scored_candidates[0][0]
            self._assign_to_cluster(state, event_id, cluster_id, vector, timestamp)
            self._append_event(state, event_id, vector, timestamp)
            self._stats["clustered_memcells"] += 1
            logger.info(
                f"[LLM Clustering] Fast path: {event_id} -> {cluster_id} "
                f"(sim={top1_sim:.3f} >= {self.config.llm_skip_threshold})"
            )
            return cluster_id, state

        # Stage 2: Fetch recent context for candidates
        cluster_context = await self._fetch_cluster_context(state, candidate_ids)

        # Stage 3: LLM decision
        clusters_json = self._build_clusters_json(
            state, candidate_ids, cluster_context
        )
        next_new_id = f"{state.next_cluster_idx:03d}"
        from memory_layer.prompts import get_prompt_by

        prompt_template = get_prompt_by("AGENT_CLUSTER_LLM_ASSIGN_PROMPT")
        prompt = prompt_template.format(
            memcell_text=text,
            clusters_json=clusters_json,
            next_new_id=next_new_id,
        )
        llm_result = await self._call_llm_for_clustering(prompt)

        if llm_result is None:
            logger.warning(
                f"[LLM Clustering] LLM call failed for event {event_id}, "
                f"falling back to embedding top-1"
            )
            # Fall back to embedding: use top-1 candidate if available, else new cluster
            if scored_candidates and scored_candidates[0][1] >= self.config.similarity_threshold:
                cluster_id = scored_candidates[0][0]
                self._assign_to_cluster(
                    state, event_id, cluster_id, vector, timestamp
                )
            else:
                cluster_id = self._create_new_cluster(
                    state, event_id, vector, timestamp, is_case=True
                )
        else:
            chosen_id = llm_result.get("cluster_id", "")
            if chosen_id in state.cluster_counts and chosen_id in state.case_cluster_ids:
                cluster_id = chosen_id
                self._assign_to_cluster(
                    state, event_id, cluster_id, vector, timestamp
                )
            else:
                cluster_id = self._create_new_cluster(
                    state, event_id, vector, timestamp, is_case=True
                )

        self._append_event(state, event_id, vector, timestamp)
        self._stats["clustered_memcells"] += 1
        reason = llm_result.get("reason", "") if llm_result else ""
        logger.info(
            f"[LLM Clustering] 🎯 Event {event_id} -> {cluster_id} "
            f"| intent: {text} | reason: {reason}"
        )
        return cluster_id, state

    def _find_top_k_clusters(
        self,
        state: MemSceneState,
        vector: Optional[np.ndarray],
        k: int = 10,
        only_cids: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """Find top-K candidate clusters by embedding similarity.

        Args:
            only_cids: If provided, only consider these cluster IDs.

        Returns:
            List of (cluster_id, similarity) tuples, sorted by similarity desc.
            Similarity is -1.0 when embedding is unavailable.
        """
        all_cids = list(state.cluster_counts.keys())
        if only_cids is not None:
            all_cids = [c for c in all_cids if c in only_cids]
        if not all_cids:
            return []

        # If no embedding or no centroids, return all with unknown similarity
        if vector is None or vector.size == 0 or not state.cluster_centroids:
            return [(c, -1.0) for c in all_cids[:k]]

        # Score each cluster by cosine similarity (ignore time gap for recall stage)
        vector_norm = np.linalg.norm(vector) + 1e-9
        scored = []
        for cid in all_cids:
            centroid = state.cluster_centroids.get(cid)
            if centroid is None or centroid.size == 0:
                scored.append((cid, -1.0))
                continue
            centroid_norm = np.linalg.norm(centroid) + 1e-9
            sim = float((centroid @ vector) / (centroid_norm * vector_norm))
            scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def _fetch_cluster_context(
        self,
        state: MemSceneState,
        candidate_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Fetch recent context texts for candidate clusters via context_fetcher.

        Returns:
            Dict mapping cluster_id -> list of recent context texts
        """
        if not self._context_fetcher or not candidate_ids:
            return {}

        max_per = self.config.llm_max_context_per_cluster

        # Collect recent event_ids per candidate cluster
        from collections import defaultdict

        candidate_set = set(candidate_ids)
        cluster_event_ids: Dict[str, List[str]] = defaultdict(list)
        for eid, cid in state.eventid_to_cluster.items():
            if cid in candidate_set:
                cluster_event_ids[cid].append(eid)

        # Take last N per cluster, collect all target event_ids
        cluster_slices: Dict[str, List[str]] = {}
        all_target_eids: List[str] = []
        for cid in candidate_ids:
            eids = cluster_event_ids.get(cid, [])
            recent = eids[-max_per:]
            cluster_slices[cid] = recent
            all_target_eids.extend(recent)

        if not all_target_eids:
            return {}

        # Call the fetcher: event_ids -> {event_id: episode_text}
        eid_to_text = await self._context_fetcher(all_target_eids)

        # Assemble per-cluster context
        result: Dict[str, List[str]] = {}
        for cid, eids in cluster_slices.items():
            texts = [eid_to_text[eid] for eid in eids if eid in eid_to_text]
            if texts:
                result[cid] = texts
        return result

    def _build_clusters_json(
        self,
        state: MemSceneState,
        candidate_ids: List[str],
        cluster_context: Dict[str, List[str]],
    ) -> str:
        """Build JSON representation of candidate clusters for LLM prompt."""
        if not candidate_ids:
            return "(No existing clusters)"

        clusters = []
        for cid in candidate_ids:
            count = state.cluster_counts.get(cid, 0)
            recent = cluster_context.get(cid, [])
            clusters.append(
                {
                    "cluster_id": cid,
                    "item_count": count,
                    "recent_task_intents": recent,
                }
            )
        return json.dumps(clusters, ensure_ascii=False, indent=2)

    async def _call_llm_for_clustering(
        self, prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Call LLM and parse clustering decision."""
        for attempt in range(3):
            try:
                resp = await self._llm_provider.generate(prompt)
                from common_utils.json_utils import parse_json_response

                data = parse_json_response(resp)
                if data and "cluster_id" in data:
                    return data
                logger.warning(
                    f"[LLM Clustering] Retry {attempt + 1}/3: invalid response format"
                )
            except Exception as e:
                logger.warning(
                    f"[LLM Clustering] Retry {attempt + 1}/3: {e}"
                )
        return None

    def _find_best_cluster(
        self,
        state: MemSceneState,
        vector: np.ndarray,
        timestamp: Optional[float],
        exclude_cids: Optional[set] = None,
    ) -> Optional[str]:
        """Find the best matching cluster for a vector."""
        if not state.cluster_centroids:
            return None

        best_similarity = -1.0
        best_cluster_id = None

        vector_norm = np.linalg.norm(vector) + 1e-9

        for cluster_id, centroid in state.cluster_centroids.items():
            if exclude_cids and cluster_id in exclude_cids:
                continue
            if centroid is None or centroid.size == 0:
                continue

            # Check time constraint
            if timestamp is not None:
                last_ts = state.cluster_last_ts.get(cluster_id)
                if last_ts is not None:
                    time_diff = abs(timestamp - last_ts)
                    if time_diff > self.config.max_time_gap_seconds:
                        continue

            # Compute cosine similarity
            centroid_norm = np.linalg.norm(centroid) + 1e-9
            similarity = float((centroid @ vector) / (centroid_norm * vector_norm))

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id

        if best_similarity >= self.config.similarity_threshold:
            return best_cluster_id

        return None

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if not self._vectorize_service:
            logger.warning("Vectorize service not available")
            return None

        try:
            vector_arr = await self._vectorize_service.get_embedding(text)
            if vector_arr is not None:
                return np.array(vector_arr, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")

        return None

    def _extract_text(self, memcell: Dict[str, Any]) -> str:
        """Extract representative text from memcell.

        Priority: clustering_text > episode > original_data
        """
        clustering_text = memcell.get("clustering_text")
        if isinstance(clustering_text, str) and clustering_text.strip():
            return clustering_text.strip()

        episode = memcell.get("episode")
        if isinstance(episode, str) and episode.strip():
            return episode.strip()

        lines = []
        original_data = memcell.get("original_data")
        if isinstance(original_data, list):
            for item in original_data:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("summary")
                    if content:
                        text = str(content).strip()
                        if text:
                            lines.append(text)

        return "\n".join(lines) if lines else str(memcell.get("event_id", ""))

    def _parse_timestamp(self, timestamp: Any) -> Optional[float]:
        """Parse timestamp to float seconds."""
        if timestamp is None:
            return None

        try:
            if isinstance(timestamp, (int, float)):
                val = float(timestamp)
                if val > 10_000_000_000:
                    val = val / 1000.0
                return val
            elif isinstance(timestamp, str):
                from common_utils.datetime_utils import from_iso_format

                dt = from_iso_format(timestamp)
                return dt.timestamp()
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")

        return None

    async def _notify_callbacks(
        self, group_id: str, memcell: Dict[str, Any], cluster_id: str
    ) -> None:
        """Notify all registered callbacks of cluster assignment."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(group_id, memcell, cluster_id)
                else:
                    callback(group_id, memcell, cluster_id)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        return dict(self._stats)
