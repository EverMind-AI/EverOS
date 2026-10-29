"""Dual-trigger contract for :func:`extract_user_profile`.

Pre-refactor, the strategy only listened for ``ProfileClusterUpdated`` — a
cluster-path-only event that ``trigger_profile_clustering`` never produces
when embedding is unavailable (Tier 1). This file pins the fix: the
strategy also listens for ``EpisodeExtracted`` (direct path) and gates each
event type via ``_profile_applies`` so exactly one path fires per memcell,
regardless of tier.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile

from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated
from everos.memory.models import Cluster, Episode, EpisodeParentType, Profile
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.strategies import EXTRACT_USER_PROFILE_SPEC, extract_user_profile
from tests.fakes import make_deps

_MODULE = "everos.memory.strategies.extract_user_profile"
_EXTRACTOR = f"{_MODULE}.ProfileExtractor"
_INTERVAL = f"{_MODULE}.PROFILE_EXTRACTION_INTERVAL"
_ALICE = Scope(
    app_id="default", project_id="default", owner_id="u_alice", owner_type="user"
)
_applies = EXTRACT_USER_PROFILE_SPEC.applies_to


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
    language: str | None = None,
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id="ep_20260517_0001",
        episode_text="alice likes hiking",
        episode_timestamp_ms=1_700_000_001_000,
        owner_id=owner_id,
        session_id="s_test",
        source=source,
        language=language,
    )


def _tier(*, embed_available: bool) -> MemoryDeps:
    return make_deps() if embed_available else make_deps(embedding=None)


def _indexed_episode(
    entry_id: str,
    parent_id: str,
    *,
    parent_type: EpisodeParentType = "memcell",
    timestamp: int = 1_700_000_001_000,
) -> Stored[Episode]:
    return Stored(
        ref=MemoryRef(id=f"u_alice_{entry_id}", entry_id=entry_id, md_path="ep.md"),
        scope=_ALICE,
        item=Episode(
            owner_id="u_alice",
            episode="...",
            summary="",
            timestamp=timestamp,
            parent_id=parent_id,
            parent_type=parent_type,
        ),
    )


def _memcell(memcell_id: str = "mc_1") -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"{memcell_id}_m1",
                role="user",
                content="hi",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            )
        ],
        timestamp=1_700_000_001_000,
    )


def _profile(**overrides: object) -> AlgoProfile:
    return AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "s",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [],
            "implicit_traits": [],
            **overrides,
        }
    )


def _cluster_with_count(cluster_id: str, count: int) -> Cluster:
    return Cluster(
        id=cluster_id,
        centroid=(0.0,) * 8,
        count=count,
        last_ts=1_700_000_001_000,
        members=[f"ep_{cluster_id}_{i}" for i in range(count)],
    )


# ── _profile_applies gate ─────────────────────────────────────────────────


def test_applies_to_cluster_event_always_true() -> None:
    """ProfileClusterUpdated only exists when trigger_profile_clustering ran
    (embed available), so accepting it unconditionally cannot double-fire."""
    for available in (True, False):
        assert _applies(_cluster_event(), _tier(embed_available=available)) is True


def test_applies_to_episode_event_only_when_no_embed() -> None:
    """Direct path fires only for pipeline-sourced episodes while embedding
    is unavailable; the cluster path owns the memcell once embed is on."""
    pipeline_event = _episode_event(source="pipeline")
    reflection_event = _episode_event(source="reflection")

    tier1 = _tier(embed_available=False)
    assert _applies(pipeline_event, tier1) is True
    assert _applies(reflection_event, tier1) is False

    tier2 = _tier(embed_available=True)
    assert _applies(pipeline_event, tier2) is False
    assert _applies(reflection_event, tier2) is False


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
    assert _applies(event_factory(), _tier(embed_available=embed_available)) is (
        expected
    )


def test_spec_registers_both_event_types() -> None:
    spec = EXTRACT_USER_PROFILE_SPEC
    assert set(spec.trigger.on) == {ProfileClusterUpdated, EpisodeExtracted}
    assert callable(spec.applies_to)


# ── direct path (Tier 1, EpisodeExtracted) ────────────────────────────────


async def test_direct_path_fetches_via_timestamp_and_writes_profile() -> None:
    """EpisodeExtracted in Tier 1 pulls memcells via the timestamp selector,
    feeds them to the LLM extractor, and persists the resulting profile."""
    deps = _tier(embed_available=False)
    deps.indexes.episode.seed(_indexed_episode("ep_1", "mc_aaaaaaaaaaa1"))
    deps.stores.memcell.seed("mc_aaaaaaaaaaa1", _memcell("mc_aaaaaaaaaaa1"))
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [{"description": "lives in tokyo"}],
            "implicit_traits": [{"description": "adventurous"}],
        }
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        await extract_user_profile(_episode_event(language="Japanese"), deps)

    extractor_call = extractor_cls.return_value.aextract.call_args
    assert [len(mc.items) for mc in extractor_call.args[0]] == [1]
    assert extractor_call.kwargs["old_profile"] is None
    assert extractor_call.kwargs["sender_id"] == "u_alice"
    assert extractor_call.kwargs["output_language"] == "Japanese"
    stored = deps.stores.profile.profiles[_ALICE]
    assert stored.summary == "Alice is a hiker."
    assert stored.primary_language == "Japanese"


async def test_cluster_path_unaffected_by_dual_trigger_refactor() -> None:
    """Cluster path (ProfileClusterUpdated): memcells are resolved from the
    cluster's members, not from the timestamp selector."""
    deps = make_deps()
    deps.stores.cluster.seed(
        _ALICE,
        "user_memory",
        Cluster(
            id="cl_user00000001",
            centroid=(0.0,) * 8,
            count=1,
            last_ts=1_700_000_001_000,
            members=["ep_20260101_0001"],
        ),
    )
    # The supplement row would be picked up only by the timestamp selector.
    deps.indexes.episode.seed(
        _indexed_episode("ep_20260101_0001", "mc_aaaaaaaaaaa1"),
        _indexed_episode("ep_unrelated", "mc_not_in_cluster"),
    )
    deps.stores.memcell.seed("mc_aaaaaaaaaaa1", _memcell("mc_aaaaaaaaaaa1"))
    deps.stores.memcell.seed("mc_not_in_cluster", _memcell("mc_not_in_cluster"))
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(
            return_value=_profile(summary="Alice is a hiker.")
        )
        await extract_user_profile(_cluster_event(), deps)

    (memcells,) = extractor_cls.return_value.aextract.call_args.args
    assert [mc.items[0].id for mc in memcells] == ["mc_aaaaaaaaaaa1_m1"]
    assert deps.stores.profile.profiles[_ALICE].summary == "Alice is a hiker."


# ── direct-path memcell selection (event-first, cascade-race-proof) ──────
#
# Round-2 review found the direct-path selector previously read the index
# exclusively — on a fresh Tier-1 install the cascade may not have indexed
# the just-arrived memcell yet, so the supplement came back empty, the
# MIN_MEMCELLS guard early-returned, and the first memory's profile was
# permanently lost. The selector now always seeds the set with
# ``event.memcell_id``; the index is a best-effort supplement.


async def _selected_memcells(deps: MemoryDeps, event: EpisodeExtracted) -> set[str]:
    """Run the direct path; return the memcell ids it asked the store for."""
    for memcell_id in (
        event.memcell_id,
        *(r.item.parent_id for r in deps.indexes.episode.rows),
    ):
        deps.stores.memcell.seed(memcell_id, _memcell(memcell_id))
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=_profile())
        await extract_user_profile(event, deps)
    (memcells,) = extractor_cls.return_value.aextract.call_args.args
    return {mc.items[0].id.removesuffix("_m1") for mc in memcells}


async def test_direct_path_returns_event_memcell_when_index_empty() -> None:
    """M4: cascade race — the index has nothing yet but the event's memcell
    is still selected, so the strategy never early-returns on the first
    memory."""
    deps = _tier(embed_available=False)
    selected = await _selected_memcells(
        deps, _episode_event(memcell_id="mc_fresh_install")
    )
    assert selected == {"mc_fresh_install"}


async def test_direct_path_returns_event_memcell_plus_supplement() -> None:
    """Union: event's memcell merged with the index supplement."""
    deps = _tier(embed_available=False)
    deps.indexes.episode.seed(
        _indexed_episode("ep_a", "mc_older_a"),
        _indexed_episode("ep_b", "mc_older_b"),
    )
    selected = await _selected_memcells(deps, _episode_event(memcell_id="mc_current"))
    assert selected == {"mc_current", "mc_older_a", "mc_older_b"}


async def test_direct_path_dedupes_when_supplement_overlaps_event() -> None:
    """No duplicate when the supplement returns the event's own memcell."""
    deps = _tier(embed_available=False)
    deps.indexes.episode.seed(_indexed_episode("ep_shared", "mc_shared"))
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=_profile())
        deps.stores.memcell.seed("mc_shared", _memcell("mc_shared"))
        await extract_user_profile(_episode_event(memcell_id="mc_shared"), deps)
    (memcells,) = extractor_cls.return_value.aextract.call_args.args
    assert len(memcells) == 1


async def test_direct_path_includes_event_memcell_even_when_timestamp_le_last_profile() -> (  # noqa: E501
    None
):
    """M5: historical-timestamp import — every indexed episode is older than
    the existing profile, so the supplement is empty; the selector must still
    include the event's memcell (matches the cluster path's
    ``c.id == event.cluster_id`` fallback)."""
    deps = _tier(embed_available=False)
    deps.stores.profile.profiles[_ALICE] = Profile(
        owner_id="u_alice", summary="prior", timestamp=2_000_000_000_000
    )
    deps.indexes.episode.seed(_indexed_episode("ep_old", "mc_old"))
    selected = await _selected_memcells(
        deps, _episode_event(memcell_id="mc_historical_import")
    )
    assert selected == {"mc_historical_import"}


# ── Tier 2+ EpisodeExtracted never double-fires ──────────────────────────


def test_tier2_episode_created_does_not_fire_direct_path() -> None:
    """For a memcell created under Tier 2+ (embed available), the direct
    path's gate must return False — the cluster path (via the later
    ProfileClusterUpdated from trigger_profile_clustering) is the only one
    that proceeds."""
    assert _applies(_episode_event(source="pipeline"), make_deps()) is False


# ── strategy-entry throttle (unified across both paths) ───────────────────
#
# The pre-refactor throttle sat inside the cluster selector so the Tier-1
# direct path silently skipped it. Bumping ``PROFILE_EXTRACTION_INTERVAL``
# to cap LLM cost therefore only slowed the cluster path down. These tests
# pin the lifted-throttle contract: both paths gate on the same modulo
# check at strategy entry, and the direct path derives its count from the
# owner's memcell episodes in the index (~1:1 with cluster.count).


def _seed_memcell_episodes(deps: MemoryDeps, count: int) -> None:
    deps.indexes.episode.seed(
        *(_indexed_episode(f"ep_{i}", f"mc_{i}") for i in range(count))
    )
    for i in range(count):
        deps.stores.memcell.seed(f"mc_{i}", _memcell(f"mc_{i}"))


async def test_direct_path_throttles_by_episode_count() -> None:
    """Tier 1: episode_count % interval != 0 skips extraction.

    6 memcell episodes, ``PROFILE_EXTRACTION_INTERVAL`` = 5 → 6 % 5 == 1 →
    gate fails → no LLM call, no profile write.
    """
    deps = _tier(embed_available=False)
    _seed_memcell_episodes(deps, 6)
    with patch(_INTERVAL, 5), patch(_EXTRACTOR) as extractor_cls:
        await extract_user_profile(_episode_event(), deps)

    extractor_cls.assert_not_called()
    assert deps.stores.profile.writes == []


async def test_direct_path_does_not_throttle_at_default_interval_1() -> None:
    """Interval=1 disables the gate outright (``interval > 1`` guard)."""
    deps = _tier(embed_available=False)
    _seed_memcell_episodes(deps, 7)
    with patch(_INTERVAL, 1), patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=_profile())
        await extract_user_profile(_episode_event(), deps)

    extractor_cls.return_value.aextract.assert_awaited_once()
    assert len(deps.stores.profile.writes) == 1


async def test_direct_path_fires_when_count_is_multiple_of_interval() -> None:
    """Count=5, interval=5 → gate passes → LLM extractor is invoked."""
    deps = _tier(embed_available=False)
    _seed_memcell_episodes(deps, 5)
    with patch(_INTERVAL, 5), patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=_profile())
        await extract_user_profile(_episode_event(), deps)

    extractor_cls.return_value.aextract.assert_awaited_once()
    assert len(deps.stores.profile.writes) == 1


async def test_cluster_path_still_throttles_after_lift() -> None:
    """Tier 2+: sum(c.count) % interval != 0 still throttles.

    Two clusters with counts 4+2 = 6, ``PROFILE_EXTRACTION_INTERVAL`` = 5 →
    6 % 5 == 1 → throttled before any memcell is selected.
    """
    deps = make_deps()
    deps.stores.cluster.seed(
        _ALICE,
        "user_memory",
        _cluster_with_count("cl_a", 4),
        _cluster_with_count("cl_b", 2),
    )
    with patch(_INTERVAL, 5), patch(_EXTRACTOR) as extractor_cls:
        await extract_user_profile(_cluster_event(), deps)

    extractor_cls.assert_not_called()
    assert deps.stores.profile.writes == []


async def test_direct_path_throttle_ignores_reflection_merged_episodes() -> None:
    """Tier 1: the throttle counter must exclude Reflection-merged rows.

    5 memcell episodes + 1 cluster-merged episode in the index: the counter
    reads 5 (not 6) — matching interval=5 exactly (5 % 5 == 0), so the gate
    passes and the LLM extractor is invoked.
    """
    deps = _tier(embed_available=False)
    _seed_memcell_episodes(deps, 5)
    deps.indexes.episode.seed(
        _indexed_episode("ep_merged", "cl_1", parent_type="cluster")
    )
    with patch(_INTERVAL, 5), patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=_profile())
        await extract_user_profile(_episode_event(), deps)

    extractor_cls.return_value.aextract.assert_awaited_once()
    assert len(deps.stores.profile.writes) == 1
