"""Dual-trigger contract for :func:`extract_user_profile`.

Pre-refactor, the strategy only listened for ``ProfileClusterUpdated`` — a
cluster-path-only event that ``trigger_profile_clustering`` never emits when
embedding is unavailable (Tier 1, see Task 14's registration gate). This
file pins the fix: the strategy now also listens for ``EpisodeExtracted``
(direct path) and gates each event type via ``_profile_applies`` so exactly
one path fires per memcell, regardless of tier.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from everalgo.types import Profile as AlgoProfile

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated
from everos.memory.strategies.extract_user_profile import (
    _profile_applies,
    _select_via_timestamp,
    extract_user_profile,
)


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _cluster_event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    cluster_id: str = "cl_user00000001",
) -> ProfileClusterUpdated:
    return ProfileClusterUpdated(
        memcell_id=memcell_id, cluster_id=cluster_id, owner_id=owner_id
    )


def _episode_event(
    *,
    owner_id: str = "u_alice",
    source: str = "pipeline",
    memcell_id: str = "mc_aaaaaaaaaaa1",
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id="ep_20260517_0001",
        episode_text="alice likes hiking",
        episode_timestamp_ms=1_700_000_001_000,
        owner_id=owner_id,
        session_id="s_test",
        source=source,
    )


def _mock_capability(*, available: bool):
    return patch(
        "everos.memory.strategies.extract_user_profile.get_embedding_capability",
        return_value=MagicMock(available=available),
    )


# ── _profile_applies gate ─────────────────────────────────────────────────


def test_applies_to_cluster_event_always_true() -> None:
    """ProfileClusterUpdated only fires when trigger_profile_clustering ran
    (embed available), so accepting it unconditionally cannot double-fire."""
    event = _cluster_event()
    for cap_available in (True, False):
        with _mock_capability(available=cap_available):
            assert _profile_applies(event) is True


def test_applies_to_episode_event_only_when_no_embed() -> None:
    """Direct path fires only for pipeline-sourced episodes while embedding
    is unavailable; the cluster path owns the memcell once embed is on."""
    pipeline_event = _episode_event(source="pipeline")
    reflection_event = _episode_event(source="reflection")

    with _mock_capability(available=False):
        assert _profile_applies(pipeline_event) is True
        assert _profile_applies(reflection_event) is False

    with _mock_capability(available=True):
        assert _profile_applies(pipeline_event) is False
        assert _profile_applies(reflection_event) is False


@pytest.mark.parametrize(
    ("event_factory", "embed_available", "expected"),
    [
        (_episode_event, False, True),
        (_episode_event, True, False),
        (_cluster_event, True, True),
        (_cluster_event, False, True),
    ],
    ids=[
        "episode+no_embed->direct_path",
        "episode+embed->skipped",
        "cluster+embed->cluster_path",
        "cluster+no_embed->cluster_path",
    ],
)
def test_applies_to_matrix(event_factory, embed_available, expected) -> None:
    with _mock_capability(available=embed_available):
        assert _profile_applies(event_factory()) is expected


def test_meta_registers_both_event_types() -> None:
    meta = extract_user_profile.meta
    assert set(meta.trigger.on) == {ProfileClusterUpdated, EpisodeExtracted}
    assert meta.applies_to is _profile_applies


# ── direct path (Tier 1, EpisodeExtracted) ────────────────────────────────


@pytest.mark.asyncio
async def test_direct_path_fetches_via_timestamp_and_writes_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EpisodeExtracted in Tier 1 pulls episodes via list_by_owner_after_ts,
    feeds them to the LLM extractor, and persists the resulting profile."""
    # `_select_via_timestamp` calls `list_by_owner_after_ts(..., columns=[...])`
    # so the repo returns raw dicts (projection contract) — not full Episode
    # objects. Mock accordingly.
    ep_row = {"parent_id": "mc_aaaaaaaaaaa1"}
    mc_row = MagicMock()
    mc_row.memcell_id = "mc_aaaaaaaaaaa1"

    from everalgo.types import ChatMessage
    from everalgo.types import MemCell as AlgoMemCell

    cell = AlgoMemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            )
        ],
        timestamp=1_700_000_001_000,
    )
    mc_row.payload_json = cell.model_dump_json()

    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": ["lives in tokyo"],
            "implicit_traits": ["adventurous"],
        }
    )

    with (
        _mock_capability(available=False),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[ep_row])
        mock_episode_repo.count_by_owner = AsyncMock(return_value=1)
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[mc_row])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_episode_event(), FakeStrategyContext())

    mock_episode_repo.list_by_owner_after_ts.assert_awaited_once_with(
        owner_id="u_alice",
        after_ts=0,
        parent_type="memcell",
        app_id="default",
        project_id="default",
        columns=["parent_id"],
    )
    mock_memcell_repo.find_by_ids.assert_awaited_once()
    assert set(mock_memcell_repo.find_by_ids.call_args.args[0]) == {"mc_aaaaaaaaaaa1"}

    extractor_call = mock_extractor_cls.return_value.aextract.call_args
    assert extractor_call.kwargs["old_profile"] is None
    assert extractor_call.kwargs["sender_id"] == "u_alice"

    write_call = mock_writer_cls.return_value.write.call_args
    assert write_call.args[0] == "u_alice"
    assert write_call.kwargs["frontmatter"].summary == "Alice is a hiker."


@pytest.mark.asyncio
async def test_cluster_path_unaffected_by_dual_trigger_refactor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster path (ProfileClusterUpdated) keeps its pre-refactor behavior:
    _select_via_cluster still drives memcell selection from cluster members."""
    import numpy as np
    from everalgo.clustering import Cluster as AlgoCluster

    cluster = AlgoCluster(
        id="cl_user00000001",
        centroid=np.zeros(1024, dtype=np.float32),
        count=1,
        last_ts=1_700_000_001_000,
        preview=[],
        members=["ep_20260101_0001"],
    )
    ep_row = MagicMock()
    ep_row.entry_id = "ep_20260101_0001"
    ep_row.parent_type = "memcell"
    ep_row.parent_id = "mc_aaaaaaaaaaa1"

    from everalgo.types import ChatMessage
    from everalgo.types import MemCell as AlgoMemCell

    cell = AlgoMemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            )
        ],
        timestamp=1_700_000_001_000,
    )
    mc_row = MagicMock()
    mc_row.memcell_id = "mc_aaaaaaaaaaa1"
    mc_row.payload_json = cell.model_dump_json()

    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_episode_repo.find_by_owner_entries = AsyncMock(return_value=[ep_row])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[mc_row])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_cluster_event(), FakeStrategyContext())

    mock_episode_repo.find_by_owner_entries.assert_awaited_once()
    mock_episode_repo.list_by_owner_after_ts.assert_not_called()
    write_call = mock_writer_cls.return_value.write.call_args
    assert write_call.kwargs["frontmatter"].summary == "Alice is a hiker."


# ── _select_via_timestamp (event-first, cascade-race-proof) ──────────────
#
# Round-2 review found the direct-path selector previously read LanceDB
# exclusively — on a fresh Tier-1 install the cascade may not have indexed
# the just-arrived memcell yet, so ``list_by_owner_after_ts`` returned
# ``[]``, the MIN_MEMCELLS guard early-returned, and the first memory's
# profile was permanently lost. The selector now always seeds the set
# with ``event.memcell_id`` (event-first, matches ``EpisodeExtracted``'s
# contract at ``events.py:40-51``); LanceDB is a best-effort supplement.


@pytest.mark.asyncio
async def test_direct_path_returns_event_memcell_when_lancedb_empty() -> None:
    """M4: cascade race — LanceDB returns [] but the event's memcell is
    still emitted, so the strategy never early-returns on the first memory."""
    event = _episode_event(memcell_id="mc_fresh_install")
    with patch(
        "everos.memory.strategies.extract_user_profile.episode_repo"
    ) as mock_episode_repo:
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[])
        result = await _select_via_timestamp(event, last_profile_ts=0)
    assert result == ["mc_fresh_install"]


@pytest.mark.asyncio
async def test_direct_path_returns_event_memcell_plus_supplement() -> None:
    """Union: event's memcell merged with the LanceDB supplement, deduped."""
    event = _episode_event(memcell_id="mc_current")
    # Projection contract: repo returns raw dicts when caller passes `columns`.
    older_a = {"parent_id": "mc_older_a"}
    older_b = {"parent_id": "mc_older_b"}
    with patch(
        "everos.memory.strategies.extract_user_profile.episode_repo"
    ) as mock_episode_repo:
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(
            return_value=[older_a, older_b]
        )
        result = await _select_via_timestamp(event, last_profile_ts=100)
    assert set(result) == {"mc_current", "mc_older_a", "mc_older_b"}
    assert len(result) == 3, "expected de-duplication, got duplicates"


@pytest.mark.asyncio
async def test_direct_path_dedupes_when_supplement_overlaps_event() -> None:
    """No duplicate when the LanceDB supplement returns the same memcell."""
    event = _episode_event(memcell_id="mc_shared")
    # Projection contract: repo returns raw dicts when caller passes `columns`.
    overlap = {"parent_id": "mc_shared"}
    with patch(
        "everos.memory.strategies.extract_user_profile.episode_repo"
    ) as mock_episode_repo:
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[overlap])
        result = await _select_via_timestamp(event, last_profile_ts=0)
    assert result == ["mc_shared"]


@pytest.mark.asyncio
async def test_direct_path_includes_event_memcell_even_when_timestamp_le_last_profile() -> (  # noqa: E501
    None
):
    """M5: historical-timestamp import — the event's own memcell has a
    timestamp <= ``last_profile_ts``, so LanceDB legitimately returns [];
    the selector must still include the event's memcell (matches the
    cluster path's ``c.id == event.cluster_id`` fallback at :139)."""
    event = _episode_event(memcell_id="mc_historical_import")
    with patch(
        "everos.memory.strategies.extract_user_profile.episode_repo"
    ) as mock_episode_repo:
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[])
        result = await _select_via_timestamp(event, last_profile_ts=2_000_000_000_000)
    assert result == ["mc_historical_import"]


# ── grep-style confirmation: Tier 2+ EpisodeExtracted never double-fires ──


def test_tier2_episode_created_does_not_fire_direct_path() -> None:
    """For a memcell created under Tier 2+ (embed available), the direct
    path's gate must return False — the cluster path (via the later
    ProfileClusterUpdated emitted by trigger_profile_clustering) is the
    only one that proceeds."""
    with _mock_capability(available=True):
        assert _profile_applies(_episode_event(source="pipeline")) is False


# ── strategy-entry throttle (unified across both paths) ───────────────────
#
# The pre-refactor throttle sat inside ``_select_via_cluster`` so the
# Tier-1 direct path silently skipped it. Bumping ``PROFILE_EXTRACTION_INTERVAL``
# to cap LLM cost therefore only slowed the cluster path down. These tests
# pin the lifted-throttle contract: both paths gate on the same modulo
# check at strategy entry, and the direct path derives its count from
# ``episode_repo.count_by_owner`` (~1:1 with cluster.count in normal usage).


def _cluster_with_count(cluster_id: str, count: int) -> object:
    """Minimal AlgoCluster stand-in — only ``count``/``last_ts``/``id`` are read."""
    import numpy as np
    from everalgo.clustering import Cluster as AlgoCluster

    return AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(1024, dtype=np.float32),
        count=count,
        last_ts=1_700_000_001_000,
        preview=[],
        members=[f"ep_{cluster_id}_{i}" for i in range(count)],
    )


@pytest.mark.asyncio
async def test_direct_path_throttles_by_episode_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 1: episode_count % interval != 0 skips extraction and logs.

    ``episode_repo.count_by_owner`` returns 6, ``PROFILE_EXTRACTION_INTERVAL``
    is 5 → 6 % 5 == 1 → gate fails → no LLM call, no writer call.
    """
    with (
        _mock_capability(available=False),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_episode_repo.count_by_owner = AsyncMock(return_value=6)
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock()
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "PROFILE_EXTRACTION_INTERVAL", 5)

        await extract_user_profile(_episode_event(), FakeStrategyContext())

    mock_episode_repo.count_by_owner.assert_awaited_once_with(
        "u_alice",
        app_id="default",
        project_id="default",
        parent_type="memcell",
    )
    mock_episode_repo.list_by_owner_after_ts.assert_not_called()
    mock_extractor_cls.return_value.aextract.assert_not_called()
    mock_writer_cls.return_value.write.assert_not_called()


@pytest.mark.asyncio
async def test_direct_path_does_not_throttle_at_default_interval_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interval=1 disables the gate outright (``interval > 1`` guard)."""
    # Projection contract: repo returns raw dicts when caller passes `columns`.
    ep_row = {"parent_id": "mc_1"}
    from everalgo.types import ChatMessage
    from everalgo.types import MemCell as AlgoMemCell

    cell = AlgoMemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            )
        ],
        timestamp=1_700_000_001_000,
    )
    mc_row = MagicMock()
    mc_row.memcell_id = "mc_1"
    mc_row.payload_json = cell.model_dump_json()
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )

    with (
        _mock_capability(available=False),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_episode_repo.count_by_owner = AsyncMock(return_value=7)
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[ep_row])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[mc_row])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "PROFILE_EXTRACTION_INTERVAL", 1)

        await extract_user_profile(_episode_event(), FakeStrategyContext())

    # Any count % 1 == 0 → gate always passes → extractor + writer both fire.
    mock_extractor_cls.return_value.aextract.assert_awaited_once()
    mock_writer_cls.return_value.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_direct_path_fires_when_count_is_multiple_of_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count=5, interval=5 → gate passes → LLM extractor is invoked."""
    # Projection contract: repo returns raw dicts when caller passes `columns`.
    ep_row = {"parent_id": "mc_1"}
    from everalgo.types import ChatMessage
    from everalgo.types import MemCell as AlgoMemCell

    cell = AlgoMemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            )
        ],
        timestamp=1_700_000_001_000,
    )
    mc_row = MagicMock()
    mc_row.memcell_id = "mc_1"
    mc_row.payload_json = cell.model_dump_json()
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )

    with (
        _mock_capability(available=False),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_episode_repo.count_by_owner = AsyncMock(return_value=5)
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[ep_row])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[mc_row])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "PROFILE_EXTRACTION_INTERVAL", 5)

        await extract_user_profile(_episode_event(), FakeStrategyContext())

    mock_extractor_cls.return_value.aextract.assert_awaited_once()
    mock_writer_cls.return_value.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_cluster_path_still_throttles_after_lift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 2+: sum(c.count) % interval != 0 still throttles.

    Preserves pre-refactor cluster-path semantics after lifting the
    throttle to strategy entry. Two clusters with counts 4+2 = 6,
    ``PROFILE_EXTRACTION_INTERVAL`` = 5 → 6 % 5 == 1 → throttled.
    """
    clusters = [
        _cluster_with_count("cl_a", 4),
        _cluster_with_count("cl_b", 2),
    ]

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=clusters)
        mock_episode_repo.find_by_owner_entries = AsyncMock(return_value=[])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock()
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "PROFILE_EXTRACTION_INTERVAL", 5)

        await extract_user_profile(_cluster_event(), FakeStrategyContext())

    mock_cluster_repo.list_for_owner.assert_awaited_once()
    mock_episode_repo.find_by_owner_entries.assert_not_called()
    mock_extractor_cls.return_value.aextract.assert_not_called()
    mock_writer_cls.return_value.write.assert_not_called()


@pytest.mark.asyncio
async def test_direct_path_throttle_ignores_reflection_merged_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 1: throttle counter must exclude Reflection-merged rows.

    ``count_by_owner`` is now passed ``parent_type='memcell'`` so it counts
    only the same rows ``_select_via_timestamp`` will later fetch. With 5
    memcell episodes + 1 cluster-merged episode in the store, the counter
    must return 5 (not 6) — matching interval=5 exactly (5 % 5 == 0), the
    gate passes and the LLM extractor is invoked.
    """
    ep_row = MagicMock()
    ep_row.parent_id = "mc_1"
    from everalgo.types import ChatMessage
    from everalgo.types import MemCell as AlgoMemCell

    cell = AlgoMemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            )
        ],
        timestamp=1_700_000_001_000,
    )
    mc_row = MagicMock()
    mc_row.memcell_id = "mc_1"
    mc_row.payload_json = cell.model_dump_json()
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )

    # Simulate the invariant the real repo now enforces: with
    # parent_type='memcell' → 5 rows; without the filter → 6 (5 memcell
    # + 1 cluster). Whichever kwarg the strategy sends decides the count.
    async def _fake_count(owner_id: str, **kwargs: object) -> int:
        return 5 if kwargs.get("parent_type") == "memcell" else 6

    with (
        _mock_capability(available=False),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
    ):
        mock_episode_repo.count_by_owner = AsyncMock(side_effect=_fake_count)
        mock_episode_repo.list_by_owner_after_ts = AsyncMock(return_value=[ep_row])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[mc_row])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "PROFILE_EXTRACTION_INTERVAL", 5)

        await extract_user_profile(_episode_event(), FakeStrategyContext())

    # Strategy passed parent_type='memcell' → fake returns 5 → 5 % 5 == 0 →
    # gate passes → extractor + writer both fire.
    mock_episode_repo.count_by_owner.assert_awaited_once_with(
        "u_alice",
        app_id="default",
        project_id="default",
        parent_type="memcell",
    )
    kwargs = mock_episode_repo.count_by_owner.await_args.kwargs
    assert kwargs["parent_type"] == "memcell"
    mock_extractor_cls.return_value.aextract.assert_awaited_once()
    mock_writer_cls.return_value.write.assert_awaited_once()
