"""Single-path contract for :func:`extract_user_profile` — decoupled from clusters.

The strategy used to have two paths. ``trigger_profile_clustering`` emitted
``ProfileClusterUpdated`` per episode and the strategy selected "every member of every
cluster fresher than the profile"; a second, direct path on :class:`EpisodeExtracted`
existed only for the no-embedding tier, and ``_profile_applies`` gated the two so
exactly one fired per memcell.

That is gone. There is one trigger (:class:`EpisodeExtracted`, ``source="pipeline"``)
and one selector (:func:`_select_via_timestamp`), so the profile is a function of the
memcell that just landed -- the same contract ``extract_episode`` and
``extract_atomic_facts`` have. This file pins that: the tests that used to say "direct
path" now describe the only path, and the ones below them assert the decoupling itself.

Clustering still runs; ``agentic`` retrieval and Reflection read ``cluster_repo``. It
just no longer gates the profile.
"""

from __future__ import annotations

import contextlib
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from everalgo.types import Profile as AlgoProfile

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted
from everos.memory.strategies.extract_user_profile import (
    _profile_applies,
    _select_via_timestamp,
    extract_user_profile,
)


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


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
    """No-op. The strategy no longer reads embedding capability -- kept so the
    existing tests still read as "this holds on either tier", which is now the
    point rather than a precondition.
    """
    del available
    return contextlib.nullcontext()


# ── the single path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetches_via_timestamp_and_writes_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EpisodeExtracted pulls episodes via list_by_owner_after_ts,
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
async def test_returns_event_memcell_when_lancedb_empty() -> None:
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
async def test_returns_event_memcell_plus_supplement() -> None:
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
async def test_dedupes_when_supplement_overlaps_event() -> None:
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
async def test_includes_event_memcell_even_when_timestamp_le_last_profile() -> None:
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
async def test_throttles_by_episode_count(
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
async def test_does_not_throttle_at_default_interval_1(
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
async def test_fires_when_count_is_multiple_of_interval(
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
async def test_throttle_ignores_reflection_merged_episodes(
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


# ── the decoupling itself ────────────────────────────────────────────────


def test_only_episode_extracted_is_registered() -> None:
    """One trigger. A second event type is what let the cluster path exist."""
    meta = extract_user_profile.meta
    assert list(meta.trigger.on) == [EpisodeExtracted]


def test_applies_regardless_of_embedding_availability() -> None:
    """The old gate stood the direct path down whenever embedding was available,
    handing the memcell to the cluster path instead. Nothing reads capability now,
    so the same data yields the same profile on every tier."""
    event = _episode_event()
    for available in (True, False):
        with patch(
            "everos.component.embedding.get_embedding_capability",
            return_value=MagicMock(available=available),
        ):
            assert _profile_applies(event) is True


def test_reflection_merged_episodes_are_still_excluded() -> None:
    """Their source memcells were merged into the profile when they first arrived."""
    assert _profile_applies(_episode_event(source="reflection")) is False


def test_strategy_module_no_longer_touches_the_cluster_repo() -> None:
    """A structural assertion on purpose: the cost of the old path was a read of the
    owner's entire cluster list plus a LanceDB fetch of every fresh cluster's members,
    per memcell. A reintroduced import brings that back silently."""
    import everos.memory.strategies.extract_user_profile as mod

    assert not hasattr(mod, "cluster_repo")
    assert not hasattr(mod, "_select_via_cluster")


def test_clustering_strategy_emits_nothing() -> None:
    """``extract_user_profile`` was its only consumer."""
    from everos.memory.strategies.trigger_profile_clustering import (
        trigger_profile_clustering,
    )

    assert trigger_profile_clustering.meta.emits == frozenset()
