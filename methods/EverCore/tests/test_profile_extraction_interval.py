"""
Tests for profile extraction interval-based throttling.

Layer 1: Unit tests (no DB, no LLM)
  - MemSceneState serialization (new memcell_info/memscene_info format)
  - Backward compatibility with old format
  - Interval skip/trigger logic
  - Timestamp-based cluster selection
  - Multi-cluster event collection
  - Participant merging

Layer 2: Persistence simulation (mock storage, no real DB)
  - State survives save/load cycles with new format
  - Restart recovery

Layer 3: End-to-end tests (mock LLM, mock DB)
  - Full flow: N memcells with interval + timestamp-based selection
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Set

from memory_layer.cluster_manager.manager import MemSceneState
from biz_layer.memorize_config import MemorizeConfig


# ============================================================================
# Layer 1: Unit Tests
# ============================================================================


class TestMemSceneStateSerialization:
    """MemSceneState to_dict/from_dict with new format."""

    def test_empty_state_roundtrip(self):
        state = MemSceneState()
        d = state.to_dict()
        assert "memcell_info" in d
        assert "memscene_info" in d
        assert "next_cluster_idx" in d
        restored = MemSceneState.from_dict(d)
        assert restored.event_ids == []
        assert restored.cluster_counts == {}

    def test_to_dict_produces_new_format(self):
        state = MemSceneState()
        state.event_ids = ["evt_1", "evt_2"]
        state.timestamps = [100.0, 200.0]
        state.eventid_to_cluster = {"evt_1": "cluster_000", "evt_2": "cluster_001"}
        state.cluster_centroids = {
            "cluster_000": np.array([1.0, 0.0], dtype=np.float32),
        }
        state.cluster_counts = {"cluster_000": 1, "cluster_001": 1}
        state.cluster_last_ts = {"cluster_000": 100.0, "cluster_001": 200.0}
        state.next_cluster_idx = 2

        d = state.to_dict()
        assert d["memcell_info"]["evt_1"] == {"memscene": "cluster_000", "timestamp": 100.0}
        assert d["memcell_info"]["evt_2"] == {"memscene": "cluster_001", "timestamp": 200.0}
        assert d["memscene_info"]["cluster_000"]["count"] == 1
        assert d["memscene_info"]["cluster_000"]["timestamp"] == 100.0
        assert d["memscene_info"]["cluster_000"]["center"] == [1.0, 0.0]
        assert d["memscene_info"]["cluster_001"]["count"] == 1
        assert d["next_cluster_idx"] == 2

    def test_roundtrip(self):
        state = MemSceneState()
        state.event_ids = ["evt_1", "evt_2"]
        state.timestamps = [100.0, 200.0]
        state.eventid_to_cluster = {"evt_1": "cluster_000", "evt_2": "cluster_000"}
        state.cluster_ids = ["cluster_000", "cluster_000"]
        state.cluster_centroids = {
            "cluster_000": np.array([1.0, 2.0], dtype=np.float32),
        }
        state.cluster_counts = {"cluster_000": 2}
        state.cluster_last_ts = {"cluster_000": 200.0}
        state.next_cluster_idx = 1

        restored = MemSceneState.from_dict(state.to_dict())
        assert restored.event_ids == ["evt_1", "evt_2"]
        assert restored.timestamps == [100.0, 200.0]
        assert restored.eventid_to_cluster == {"evt_1": "cluster_000", "evt_2": "cluster_000"}
        assert restored.cluster_counts == {"cluster_000": 2}
        assert restored.cluster_last_ts == {"cluster_000": 200.0}
        np.testing.assert_array_almost_equal(
            restored.cluster_centroids["cluster_000"], [1.0, 2.0]
        )

    def test_backward_compat_old_format(self):
        """Old-format data (without memcell_info) should still load."""
        old_data = {
            "event_ids": ["evt_1"],
            "timestamps": [100.0],
            "cluster_ids": ["cluster_000"],
            "eventid_to_cluster": {"evt_1": "cluster_000"},
            "next_cluster_idx": 1,
            "cluster_centroids": {"cluster_000": [1.0, 0.0]},
            "cluster_counts": {"cluster_000": 1},
            "cluster_last_ts": {"cluster_000": 100.0},
        }
        state = MemSceneState.from_dict(old_data)
        assert state.event_ids == ["evt_1"]
        assert state.eventid_to_cluster == {"evt_1": "cluster_000"}
        assert state.cluster_counts == {"cluster_000": 1}
        assert state.cluster_last_ts == {"cluster_000": 100.0}

    def test_from_dict_case_cluster_ids_none(self):
        """case_cluster_ids=None in DB should deserialize to empty set."""
        data = {
            "memcell_info": {},
            "memscene_info": {},
            "next_cluster_idx": 0,
            "case_cluster_ids": None,
        }
        state = MemSceneState.from_dict(data)
        assert state.case_cluster_ids == set()

    def test_from_dict_case_cluster_ids_missing(self):
        """Missing case_cluster_ids key should deserialize to empty set."""
        data = {
            "memcell_info": {},
            "memscene_info": {},
            "next_cluster_idx": 0,
        }
        state = MemSceneState.from_dict(data)
        assert state.case_cluster_ids == set()

    def test_from_dict_case_cluster_ids_with_values(self):
        """case_cluster_ids with values should roundtrip correctly."""
        state = MemSceneState()
        state.case_cluster_ids = {"cluster_000", "cluster_001"}
        d = state.to_dict()
        restored = MemSceneState.from_dict(d)
        assert restored.case_cluster_ids == {"cluster_000", "cluster_001"}


class TestIntervalLogic:
    """Interval skip/trigger decision logic."""

    @staticmethod
    def _should_extract(config: MemorizeConfig, total_count: int) -> bool:
        return (
            config.profile_extraction_interval <= 1
            or total_count % config.profile_extraction_interval == 0
        )

    def test_interval_1_always_triggers(self):
        config = MemorizeConfig(profile_extraction_interval=1)
        for count in range(1, 10):
            assert self._should_extract(config, count) is True

    def test_interval_3_triggers_on_multiples(self):
        config = MemorizeConfig(profile_extraction_interval=3)
        results = {i: self._should_extract(config, i) for i in range(1, 10)}
        assert results == {
            1: False, 2: False, 3: True,
            4: False, 5: False, 6: True,
            7: False, 8: False, 9: True,
        }

    def test_interval_0_treated_as_always(self):
        config = MemorizeConfig(profile_extraction_interval=0)
        assert self._should_extract(config, 1) is True
        assert self._should_extract(config, 5) is True


class TestTimestampBasedSelection:
    """Timestamp-based cluster selection for profile extraction."""

    @staticmethod
    def _select_clusters(
        cluster_last_ts: Dict[str, float],
        last_profile_ts: float,
        current_cluster_id: str,
    ) -> List[str]:
        """Replicate the logic from mem_memorize.py."""
        target = [
            cid for cid, ts in cluster_last_ts.items()
            if ts is not None and ts > last_profile_ts
        ]
        if current_cluster_id not in target:
            target.append(current_cluster_id)
        return target

    def test_first_extraction_selects_all(self):
        """When last_profile_ts=0 (no profile yet), all clusters selected."""
        result = self._select_clusters(
            cluster_last_ts={"c0": 100.0, "c1": 200.0, "c2": 300.0},
            last_profile_ts=0.0,
            current_cluster_id="c2",
        )
        assert sorted(result) == ["c0", "c1", "c2"]

    def test_only_new_clusters_selected(self):
        """Only clusters updated after last profile extraction."""
        result = self._select_clusters(
            cluster_last_ts={"c0": 100.0, "c1": 200.0, "c2": 300.0},
            last_profile_ts=150.0,
            current_cluster_id="c2",
        )
        assert sorted(result) == ["c1", "c2"]

    def test_current_cluster_always_included(self):
        """Current cluster is always included even if its timestamp is old."""
        result = self._select_clusters(
            cluster_last_ts={"c0": 100.0, "c1": 50.0},
            last_profile_ts=150.0,
            current_cluster_id="c1",
        )
        assert "c1" in result

    def test_all_old_returns_only_current(self):
        """When all clusters are older than last_profile_ts, only current."""
        result = self._select_clusters(
            cluster_last_ts={"c0": 100.0, "c1": 50.0},
            last_profile_ts=200.0,
            current_cluster_id="c0",
        )
        assert result == ["c0"]

    def test_none_timestamps_excluded(self):
        result = self._select_clusters(
            cluster_last_ts={"c0": None, "c1": 200.0},
            last_profile_ts=0.0,
            current_cluster_id="c0",
        )
        assert sorted(result) == ["c0", "c1"]


class TestMultiClusterEventCollection:
    """Collecting event_ids from multiple clusters."""

    def test_collects_from_all_target_clusters(self):
        state = MemSceneState()
        state.eventid_to_cluster = {
            "evt_1": "cluster_000",
            "evt_2": "cluster_000",
            "evt_3": "cluster_001",
            "evt_4": "cluster_002",
            "evt_5": "cluster_001",
        }
        target_set = {"cluster_000", "cluster_001"}
        current_event_id = "evt_1"

        result = {
            eid for eid, cid in state.eventid_to_cluster.items()
            if cid in target_set and eid != current_event_id
        }

        assert result == {"evt_2", "evt_3", "evt_5"}

    def test_excludes_current_event(self):
        state = MemSceneState()
        state.eventid_to_cluster = {"evt_1": "cluster_000"}
        target_set = {"cluster_000"}

        result = {
            eid for eid, cid in state.eventid_to_cluster.items()
            if cid in target_set and eid != "evt_1"
        }

        assert result == set()

    def test_empty_target_returns_empty(self):
        state = MemSceneState()
        state.eventid_to_cluster = {"evt_1": "cluster_000"}

        result = {
            eid for eid, cid in state.eventid_to_cluster.items()
            if cid in set() and eid != "evt_1"
        }

        assert result == set()


class TestParticipantMerging:
    """Participant deduplication from multiple memcells."""

    @staticmethod
    def _merge_participants(memcells: list) -> List[str]:
        all_participants: Set[str] = set()
        for mc in memcells:
            participants = (
                mc.participants
                if hasattr(mc, "participants")
                else mc.get("participants", [])
            )
            all_participants.update(participants or [])
        return [
            u for u in all_participants
            if "robot" not in u.lower() and "assistant" not in u.lower()
        ]

    def test_merges_and_deduplicates(self):
        memcells = [
            MagicMock(participants=["user_a", "user_b", "robot_1"]),
            MagicMock(participants=["user_b", "user_c", "assistant"]),
            MagicMock(participants=["user_a", "user_d"]),
        ]
        result = sorted(self._merge_participants(memcells))
        assert result == ["user_a", "user_b", "user_c", "user_d"]

    def test_handles_none_participants(self):
        memcells = [
            MagicMock(participants=None),
            MagicMock(participants=["user_a"]),
        ]
        assert self._merge_participants(memcells) == ["user_a"]

    def test_all_robots_returns_empty(self):
        memcells = [
            MagicMock(participants=["robot_1", "Assistant"]),
        ]
        assert self._merge_participants(memcells) == []


# ============================================================================
# Layer 2: Persistence Simulation (mock storage, no real DB)
# ============================================================================


class InMemoryMemSceneStorage:
    """In-memory mock of MemSceneRawRepository for testing."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def save_mem_scene(self, group_id: str, state: Dict[str, Any]) -> bool:
        self._store[group_id] = state
        return True

    async def load_mem_scene(self, group_id: str) -> Any:
        return self._store.get(group_id)

    async def clear(self, group_id: str) -> bool:
        self._store.pop(group_id, None)
        return True


class TestStatePersistence:
    """Verify MemSceneState survives save/load cycles with new format."""

    @pytest.fixture
    def storage(self):
        return InMemoryMemSceneStorage()

    @pytest.mark.asyncio
    async def test_save_and_load_new_format(self, storage):
        group_id = "__test_new_format__"

        state = MemSceneState()
        state.event_ids = ["evt_1", "evt_2"]
        state.timestamps = [100.0, 200.0]
        state.eventid_to_cluster = {"evt_1": "cluster_000", "evt_2": "cluster_001"}
        state.cluster_ids = ["cluster_000", "cluster_001"]
        state.cluster_counts = {"cluster_000": 1, "cluster_001": 1}
        state.cluster_last_ts = {"cluster_000": 100.0, "cluster_001": 200.0}
        state.next_cluster_idx = 2

        await storage.save_mem_scene(group_id, state.to_dict())
        loaded_dict = await storage.load_mem_scene(group_id)

        # Verify new format
        assert "memcell_info" in loaded_dict
        assert "memscene_info" in loaded_dict

        loaded_state = MemSceneState.from_dict(loaded_dict)
        assert loaded_state.cluster_counts == {"cluster_000": 1, "cluster_001": 1}
        assert loaded_state.eventid_to_cluster == {"evt_1": "cluster_000", "evt_2": "cluster_001"}

    @pytest.mark.asyncio
    async def test_restart_recovery(self, storage):
        """Simulate server restart: state survives in storage."""
        group_id = "__test_restart__"

        state = MemSceneState()
        state.event_ids = ["evt_1"]
        state.timestamps = [100.0]
        state.eventid_to_cluster = {"evt_1": "cluster_000"}
        state.cluster_ids = ["cluster_000"]
        state.cluster_counts = {"cluster_000": 1}
        state.cluster_last_ts = {"cluster_000": 100.0}
        state.next_cluster_idx = 1
        await storage.save_mem_scene(group_id, state.to_dict())

        # "Restart": load from storage
        fresh_dict = await storage.load_mem_scene(group_id)
        fresh_state = MemSceneState.from_dict(fresh_dict)

        assert fresh_state.event_ids == ["evt_1"]
        assert fresh_state.cluster_counts == {"cluster_000": 1}
        assert fresh_state.next_cluster_idx == 1


# ============================================================================
# Layer 3: End-to-End Tests (mock LLM + DB)
# ============================================================================


class TestProfileExtractionE2E:
    """End-to-end: verify extraction call count with timestamp-based selection.

    Simulates the caller logic from mem_memorize.py using timestamp comparison
    instead of pending_profile_cluster_ids.
    """

    @staticmethod
    def _simulate_flow(
        memcell_clusters: List[str],
        memcell_timestamps: List[float],
        interval: int,
    ) -> List[List[str]]:
        """Simulate the caller logic from mem_memorize.py.

        Args:
            memcell_clusters: cluster_id assigned to each incoming memcell.
            memcell_timestamps: timestamp of each incoming memcell.
            interval: profile_extraction_interval config value.

        Returns:
            List of cluster_ids lists passed to each extraction call.
        """
        config = MemorizeConfig(profile_extraction_interval=interval)
        cluster_counts: Dict[str, int] = {}
        cluster_last_ts: Dict[str, float] = {}
        last_profile_ts = 0.0
        extraction_calls = []

        for cluster_id, ts in zip(memcell_clusters, memcell_timestamps):
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            cluster_last_ts[cluster_id] = max(
                cluster_last_ts.get(cluster_id, 0.0), ts
            )

            total = sum(cluster_counts.values())
            should_extract = (
                config.profile_extraction_interval <= 1
                or total % config.profile_extraction_interval == 0
            )

            if should_extract:
                target = [
                    cid for cid, c_ts in cluster_last_ts.items()
                    if c_ts > last_profile_ts
                ]
                if cluster_id not in target:
                    target.append(cluster_id)
                extraction_calls.append(sorted(target))
                last_profile_ts = ts  # profile updated at this point

        return extraction_calls

    def test_interval_1_extracts_every_time(self):
        calls = self._simulate_flow(
            ["c0", "c1", "c0", "c1", "c2"],
            [100, 200, 300, 400, 500],
            interval=1,
        )
        assert len(calls) == 5

    def test_interval_3_reduces_calls(self):
        calls = self._simulate_flow(
            ["c0", "c1", "c0", "c1", "c2", "c0"],
            [100, 200, 300, 400, 500, 600],
            interval=3,
        )
        # total=3 triggers at memcell 3, total=6 triggers at memcell 6
        assert len(calls) == 2

    def test_first_extraction_includes_all_clusters(self):
        calls = self._simulate_flow(
            ["c0", "c1", "c2"],
            [100, 200, 300],
            interval=3,
        )
        assert len(calls) == 1
        assert calls[0] == ["c0", "c1", "c2"]

    def test_second_extraction_only_new_clusters(self):
        calls = self._simulate_flow(
            ["c0", "c1", "c2", "c3", "c0", "c3"],
            [100, 200, 300, 400, 500, 600],
            interval=3,
        )
        # First extraction at memcell 3 (ts=300): all [c0, c1, c2], last_profile_ts=300
        # Second extraction at memcell 6 (ts=600): c3(400>300), c0(500>300), c3(600>300)
        assert len(calls) == 2
        assert calls[0] == ["c0", "c1", "c2"]
        assert calls[1] == ["c0", "c3"]

    def test_no_cluster_missed(self):
        """Every cluster with new data must appear in at least one extraction call."""
        memcell_clusters = ["c0", "c1", "c2", "c3", "c0", "c1"]
        calls = self._simulate_flow(
            memcell_clusters,
            [100, 200, 300, 400, 500, 600],
            interval=3,
        )
        all_in_calls = set()
        for call in calls:
            all_in_calls.update(call)
        assert all_in_calls == set(memcell_clusters)
