"""Unit tests for ClusterManager._cluster_memcell_llm.

Covers every branch of the LLM-based clustering method:
1. Missing event_id
2. No LLM provider (embedding fallback)
   2a. top-1 sim >= threshold -> assign existing
   2b. top-1 sim < threshold -> new cluster
3. No existing case clusters -> new cluster
4. Fast path: top-1 sim >= llm_skip_threshold
5. LLM failure (returns None) -> embedding fallback
   5a. top-1 sim >= threshold -> assign existing
   5b. no good candidate -> new cluster
6. LLM returns valid result
   6a. chosen_id is valid case cluster -> assign
   6b. chosen_id invalid -> new cluster
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from memory_layer.cluster_manager.manager import ClusterManager, MemSceneState
from memory_layer.cluster_manager.config import ClusterManagerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> ClusterManagerConfig:
    defaults = dict(
        similarity_threshold=0.65,
        llm_skip_threshold=0.85,
        llm_top_k_clusters=5,
        llm_max_context_per_cluster=3,
    )
    defaults.update(overrides)
    return ClusterManagerConfig(**defaults)


def _make_manager(
    config=None,
    llm_provider=None,
    context_fetcher=None,
    embedding_vec=None,
) -> ClusterManager:
    """Build a ClusterManager with vectorize service mocked out."""
    cfg = config or _make_config()
    mgr = ClusterManager.__new__(ClusterManager)
    mgr.config = cfg
    mgr._callbacks = []
    mgr._llm_provider = llm_provider
    mgr._context_fetcher = context_fetcher
    mgr._stats = {
        "total_memcells": 0,
        "clustered_memcells": 0,
        "new_clusters": 0,
        "failed_embeddings": 0,
    }

    # Mock vectorize service to return a controlled vector
    mock_vs = AsyncMock()
    if embedding_vec is not None:
        mock_vs.get_embedding = AsyncMock(return_value=embedding_vec.tolist())
    else:
        mock_vs.get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])
    mgr._vectorize_service = mock_vs
    return mgr


def _make_memcell(event_id="evt_1", text="some task", timestamp=1000.0):
    return {
        "event_id": event_id,
        "clustering_text": text,
        "timestamp": timestamp,
    }


def _state_with_case_cluster(
    cluster_id="cluster_000",
    event_id="existing_evt",
    centroid=None,
    count=1,
):
    """Build a MemSceneState that already has one case cluster."""
    state = MemSceneState()
    state.next_cluster_idx = 1
    state.case_cluster_ids = {cluster_id}
    state.cluster_counts[cluster_id] = count
    state.cluster_last_ts[cluster_id] = 900.0
    state.eventid_to_cluster[event_id] = cluster_id
    state.event_ids.append(event_id)
    state.timestamps.append(900.0)
    if centroid is not None:
        state.cluster_centroids[cluster_id] = centroid
        state.vectors.append(centroid)
    else:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state.cluster_centroids[cluster_id] = vec
        state.vectors.append(vec)
    state.cluster_ids.append(cluster_id)
    return state


# ===========================================================================
# 1. Missing event_id -> (None, state)
# ===========================================================================


class TestMissingEventId:

    @pytest.mark.asyncio
    async def test_empty_event_id_returns_none(self):
        mgr = _make_manager(llm_provider=MagicMock())
        state = MemSceneState()
        memcell = {"event_id": "", "clustering_text": "x", "timestamp": 1.0}

        cid, out_state = await mgr._cluster_memcell_llm(memcell, state)

        assert cid is None
        assert out_state is state
        assert mgr._stats["total_memcells"] == 1
        assert mgr._stats["clustered_memcells"] == 0

    @pytest.mark.asyncio
    async def test_missing_event_id_key_returns_none(self):
        mgr = _make_manager(llm_provider=MagicMock())
        state = MemSceneState()
        memcell = {"clustering_text": "x"}

        cid, _ = await mgr._cluster_memcell_llm(memcell, state)
        assert cid is None


# ===========================================================================
# 2. No LLM provider -> embedding fallback
# ===========================================================================


class TestNoLlmProvider:

    @pytest.mark.asyncio
    async def test_no_llm_assign_existing_when_similar(self):
        """2a: top-1 sim >= threshold -> assign to existing cluster."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mgr = _make_manager(
            llm_provider=None,
            embedding_vec=centroid,  # identical to centroid -> sim=1.0
        )
        state = _state_with_case_cluster(centroid=centroid)

        cid, out_state = await mgr._cluster_memcell_llm(
            _make_memcell(), state
        )

        assert cid == "cluster_000"
        assert "evt_1" in out_state.eventid_to_cluster
        assert out_state.eventid_to_cluster["evt_1"] == "cluster_000"
        assert mgr._stats["clustered_memcells"] == 1

    @pytest.mark.asyncio
    async def test_no_llm_new_cluster_when_dissimilar(self):
        """2b: top-1 sim < threshold -> new cluster."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        orthogonal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        mgr = _make_manager(
            llm_provider=None,
            embedding_vec=orthogonal,  # sim ~ 0 with centroid
        )
        state = _state_with_case_cluster(centroid=centroid)

        cid, out_state = await mgr._cluster_memcell_llm(
            _make_memcell(), state
        )

        assert cid != "cluster_000"
        assert cid.startswith("cluster_")
        assert cid in out_state.case_cluster_ids
        assert mgr._stats["new_clusters"] == 1


# ===========================================================================
# 3. No existing case clusters -> create first case cluster
# ===========================================================================


class TestNoCaseClusters:

    @pytest.mark.asyncio
    async def test_first_case_cluster_created(self):
        mgr = _make_manager(llm_provider=MagicMock())
        state = MemSceneState()

        cid, out_state = await mgr._cluster_memcell_llm(
            _make_memcell(), state
        )

        assert cid == "cluster_000"
        assert cid in out_state.case_cluster_ids
        assert out_state.eventid_to_cluster["evt_1"] == cid
        assert mgr._stats["new_clusters"] == 1
        assert mgr._stats["clustered_memcells"] == 1
        assert "evt_1" in out_state.event_ids


# ===========================================================================
# 4. Fast path: top-1 sim >= llm_skip_threshold
# ===========================================================================


class TestFastPath:

    @pytest.mark.asyncio
    async def test_skip_llm_when_very_similar(self):
        """sim=1.0 >= llm_skip_threshold=0.85 -> assign without LLM."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_llm = MagicMock()
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=centroid,
        )
        state = _state_with_case_cluster(centroid=centroid)

        cid, out_state = await mgr._cluster_memcell_llm(
            _make_memcell(), state
        )

        assert cid == "cluster_000"
        assert out_state.eventid_to_cluster["evt_1"] == "cluster_000"
        assert mgr._stats["clustered_memcells"] == 1
        # LLM should NOT have been called
        assert not hasattr(mock_llm, 'generate') or not mock_llm.generate.called

    @pytest.mark.asyncio
    async def test_no_fast_path_when_below_threshold(self):
        """sim < llm_skip_threshold -> should proceed to LLM stage."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        different = np.array([0.6, 0.8, 0.0], dtype=np.float32)  # sim ~ 0.6
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value='{"cluster_id": "cluster_000", "reason": "same topic"}'
        )
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=different,
        )
        state = _state_with_case_cluster(centroid=centroid)

        with patch(
            "memory_layer.prompts.get_prompt_by",
            return_value="{memcell_text}{clusters_json}{next_new_id}",
        ):
            cid, _ = await mgr._cluster_memcell_llm(
                _make_memcell(), state
            )

        # LLM was called and returned cluster_000
        assert cid == "cluster_000"
        mock_llm.generate.assert_called()


# ===========================================================================
# 5. LLM failure (returns None) -> embedding fallback
# ===========================================================================


class TestLlmFailureFallback:

    @pytest.mark.asyncio
    async def test_llm_fail_assign_existing_when_similar(self):
        """5a: LLM fails, top-1 sim >= threshold -> assign existing."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        close_vec = np.array([0.95, 0.31, 0.0], dtype=np.float32)  # sim ~ 0.95
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="invalid json {{{")
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=close_vec,
            config=_make_config(llm_skip_threshold=1.0),  # never skip
        )
        state = _state_with_case_cluster(centroid=centroid)

        with patch(
            "memory_layer.prompts.get_prompt_by",
            return_value="{memcell_text}{clusters_json}{next_new_id}",
        ):
            cid, out_state = await mgr._cluster_memcell_llm(
                _make_memcell(), state
            )

        assert cid == "cluster_000"
        assert out_state.eventid_to_cluster["evt_1"] == "cluster_000"

    @pytest.mark.asyncio
    async def test_llm_fail_new_cluster_when_dissimilar(self):
        """5b: LLM fails, no good candidate -> new cluster."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        orthogonal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="invalid json {{{")
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=orthogonal,
            config=_make_config(llm_skip_threshold=1.0),
        )
        state = _state_with_case_cluster(centroid=centroid)

        with patch(
            "memory_layer.prompts.get_prompt_by",
            return_value="{memcell_text}{clusters_json}{next_new_id}",
        ):
            cid, out_state = await mgr._cluster_memcell_llm(
                _make_memcell(), state
            )

        assert cid != "cluster_000"
        assert cid in out_state.case_cluster_ids
        assert mgr._stats["new_clusters"] == 1


# ===========================================================================
# 6. LLM returns valid result
# ===========================================================================


class TestLlmValidResult:

    @pytest.mark.asyncio
    async def test_llm_assigns_valid_existing_cluster(self):
        """6a: LLM returns valid case cluster_id -> assign."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        different = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value='{"cluster_id": "cluster_000", "reason": "related"}'
        )
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=different,
            config=_make_config(llm_skip_threshold=1.0),
        )
        state = _state_with_case_cluster(centroid=centroid)

        with patch(
            "memory_layer.prompts.get_prompt_by",
            return_value="{memcell_text}{clusters_json}{next_new_id}",
        ):
            cid, out_state = await mgr._cluster_memcell_llm(
                _make_memcell(), state
            )

        assert cid == "cluster_000"
        assert out_state.eventid_to_cluster["evt_1"] == "cluster_000"
        assert mgr._stats["clustered_memcells"] == 1

    @pytest.mark.asyncio
    async def test_llm_returns_new_cluster_id(self):
        """6b: LLM returns an id not in state -> new cluster."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        different = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value='{"cluster_id": "001", "reason": "new topic"}'
        )
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=different,
            config=_make_config(llm_skip_threshold=1.0),
        )
        state = _state_with_case_cluster(centroid=centroid)

        with patch(
            "memory_layer.prompts.get_prompt_by",
            return_value="{memcell_text}{clusters_json}{next_new_id}",
        ):
            cid, out_state = await mgr._cluster_memcell_llm(
                _make_memcell(), state
            )

        assert cid != "cluster_000"
        assert cid in out_state.case_cluster_ids
        assert mgr._stats["new_clusters"] == 1

    @pytest.mark.asyncio
    async def test_llm_returns_non_case_cluster_creates_new(self):
        """6b variant: LLM returns cluster_id that exists but is NOT a case cluster."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        different = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mock_llm = AsyncMock()
        # Return a cluster id that exists in cluster_counts but not in case_cluster_ids
        mock_llm.generate = AsyncMock(
            return_value='{"cluster_id": "cluster_999", "reason": "matched"}'
        )
        mgr = _make_manager(
            llm_provider=mock_llm,
            embedding_vec=different,
            config=_make_config(llm_skip_threshold=1.0),
        )
        state = _state_with_case_cluster(centroid=centroid)
        # Add a non-case cluster
        state.cluster_counts["cluster_999"] = 2

        with patch(
            "memory_layer.prompts.get_prompt_by",
            return_value="{memcell_text}{clusters_json}{next_new_id}",
        ):
            cid, out_state = await mgr._cluster_memcell_llm(
                _make_memcell(), state
            )

        # Should NOT assign to cluster_999 because it's not a case cluster
        assert cid != "cluster_999"
        assert cid in out_state.case_cluster_ids


# ===========================================================================
# State mutation correctness
# ===========================================================================


class TestStateMutation:

    @pytest.mark.asyncio
    async def test_event_appended_to_state(self):
        """Every successful path appends event to state lists."""
        mgr = _make_manager(llm_provider=MagicMock())
        state = MemSceneState()

        cid, out_state = await mgr._cluster_memcell_llm(
            _make_memcell(event_id="e1", timestamp=500.0), state
        )

        assert "e1" in out_state.event_ids
        assert 500.0 in out_state.timestamps
        assert len(out_state.vectors) == 1

    @pytest.mark.asyncio
    async def test_stats_incremented(self):
        """total_memcells and clustered_memcells always incremented on success."""
        mgr = _make_manager(llm_provider=MagicMock())
        state = MemSceneState()

        await mgr._cluster_memcell_llm(_make_memcell(), state)

        assert mgr._stats["total_memcells"] == 1
        assert mgr._stats["clustered_memcells"] == 1

    @pytest.mark.asyncio
    async def test_multiple_memcells_increment_cluster_count(self):
        """Assigning two events to the same cluster increments count."""
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mgr = _make_manager(llm_provider=None, embedding_vec=centroid)
        state = _state_with_case_cluster(centroid=centroid)

        original_count = state.cluster_counts["cluster_000"]
        await mgr._cluster_memcell_llm(_make_memcell(), state)

        assert state.cluster_counts["cluster_000"] > original_count
