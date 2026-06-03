"""Tests for :func:`extract_user_profile`.

Heavy mocking — the strategy threads through ``cluster_repo`` (sqlite),
``memcell_repo`` (sqlite, payload deserialise), ``ProfileReader`` /
``ProfileWriter`` (md), and ``ProfileExtractor`` (algo). We mock all
seams so the test exercises the orchestration only.
"""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile

from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import UserProfileFrontmatter
from everos.memory.events import ProfileClusterUpdated
from everos.memory.strategies._partition_locks import _reset_for_tests
from everos.memory.strategies.extract_user_profile import extract_user_profile


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    cluster_id: str = "cl_user00000001",
) -> ProfileClusterUpdated:
    return ProfileClusterUpdated(
        memcell_id=memcell_id,
        cluster_id=cluster_id,
        owner_id=owner_id,
    )


def _algo_cluster(*, cluster_id: str, members: list[str], last_ts: int) -> AlgoCluster:
    return AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(1024, dtype=np.float32),
        count=len(members),
        last_ts=last_ts,
        preview=[],
        members=members,
    )


def _memcell_row(memcell_id: str, *, sender_id: str, ts_ms: int) -> MagicMock:
    """Stand-in for a sqlite Memcell row — only ``payload_json`` is read."""
    cell = MemCell(
        items=[
            ChatMessage(
                id=f"{memcell_id}_m1",
                role="user",
                content=f"hi from {sender_id}",
                timestamp=ts_ms,
                sender_id=sender_id,
            ),
        ],
        timestamp=ts_ms,
    )
    row = MagicMock()
    row.memcell_id = memcell_id
    row.payload_json = cell.model_dump_json()
    return row


async def test_strategy_meta_is_attached() -> None:
    meta = extract_user_profile._ome_strategy_meta  # type: ignore[attr-defined]
    assert meta.name == "extract_user_profile"
    assert ProfileClusterUpdated in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 2


@pytest.mark.asyncio
async def test_init_mode_writes_profile_when_no_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No prior profile → ProfileExtractor invoked without ``old_profile``."""
    cluster = _algo_cluster(
        cluster_id="cl_user00000001",
        members=["mc_aaaaaaaaaaa1"],
        last_ts=1_700_000_001_000,
    )
    rows = [
        _memcell_row("mc_aaaaaaaaaaa1", sender_id="u_alice", ts_ms=1_700_000_001_000)
    ]
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
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
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
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(), FakeStrategyContext())

    # INIT mode — old_profile is None.
    extractor_call = mock_extractor_cls.return_value.aextract.call_args
    assert extractor_call.kwargs["old_profile"] is None
    assert extractor_call.kwargs["sender_id"] == "u_alice"
    assert [mc.timestamp for mc in extractor_call.args[0]] == [1_700_000_001_000]

    # Writer received the freshly built frontmatter.
    write_call = mock_writer_cls.return_value.write.call_args
    assert write_call.args[0] == "u_alice"
    fm = write_call.kwargs["frontmatter"]
    assert fm.user_id == "u_alice"
    assert fm.summary == "Alice is a hiker."
    assert fm.profile_timestamp_ms == 1_700_000_001_000
    assert fm.explicit_info == ["lives in tokyo"]
    assert fm.implicit_traits == ["adventurous"]
    assert write_call.kwargs["body"] == "Alice is a hiker."


@pytest.mark.asyncio
async def test_update_mode_rehydrates_old_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing profile → algo Profile rehydrated and passed as old_profile."""
    cluster = _algo_cluster(
        cluster_id="cl_user00000001",
        members=["mc_aaaaaaaaaaa1"],
        last_ts=1_700_000_002_000,
    )
    rows = [
        _memcell_row("mc_aaaaaaaaaaa1", sender_id="u_alice", ts_ms=1_700_000_002_000)
    ]
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="prior summary",
        explicit_info=["prior fact"],
        implicit_traits=["prior trait"],
        profile_timestamp_ms=1_700_000_000_000,
    )
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "updated summary",
            "timestamp": 1_700_000_002_000,
            "explicit_info": ["prior fact", "new fact"],
            "implicit_traits": ["prior trait"],
        }
    )

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
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
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=rows)
        mock_reader_cls.return_value.read = AsyncMock(
            return_value=(existing_fm, "prior summary")
        )
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(_event(), FakeStrategyContext())

    # UPDATE mode — old_profile is the rehydrated algo type carrying prior fields.
    extractor_call = mock_extractor_cls.return_value.aextract.call_args
    old = extractor_call.kwargs["old_profile"]
    assert isinstance(old, AlgoProfile)
    assert old.summary == "prior summary"
    assert old.timestamp == 1_700_000_000_000
    assert old.model_dump()["explicit_info"] == ["prior fact"]


@pytest.mark.asyncio
async def test_skips_when_no_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty target cluster set (no fresh clusters) → no extractor call."""
    # Existing profile timestamp newer than every cluster's last_ts → no
    # target_cluster matches `last_ts > last_profile_ts`, but the current
    # cluster_id should still force inclusion. Set the current cluster id
    # to a non-existent value to drop everything.
    stale_cluster = _algo_cluster(
        cluster_id="cl_other000001",
        members=["mc_other00000"],
        last_ts=1_600_000_000_000,
    )
    existing_fm = UserProfileFrontmatter(
        id="profile_u_alice",
        user_id="u_alice",
        summary="prior",
        explicit_info=[],
        implicit_traits=[],
        profile_timestamp_ms=1_900_000_000_000,
    )

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
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
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[stale_cluster])
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[])
        mock_reader_cls.return_value.read = AsyncMock(
            return_value=(existing_fm, "prior")
        )
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock()
        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_reader", None, raising=False)

        await extract_user_profile(
            _event(cluster_id="cl_unknown00000"), FakeStrategyContext()
        )

    mock_extractor_cls.return_value.aextract.assert_not_called()
    mock_writer_cls.return_value.write.assert_not_called()


# ── partition lock (owner_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(
    owner_a: str, owner_b: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Drive two extract_user_profile runs and record entry/exit order."""
    log: list[str] = []

    async def mock_aextract(_memcells, *, sender_id, **_kwargs):
        log.append(f"enter:{sender_id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{sender_id}")
        return AlgoProfile(
            owner_id=sender_id,
            summary="summary",
            timestamp=1_700_000_000_000,
            explicit_info=[],
            implicit_traits=[],
        )

    cluster_a = _algo_cluster(
        cluster_id="cl_a", members=["mc_a"], last_ts=1_700_000_000_000
    )
    cluster_b = _algo_cluster(
        cluster_id="cl_b", members=["mc_b"], last_ts=1_700_000_000_000
    )

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            side_effect=lambda owner, _kind, **_kw: (
                [cluster_a] if owner == owner_a else [cluster_b]
            )
        )
        mock_memcell_repo.find_by_ids = AsyncMock(
            side_effect=lambda ids: [
                _memcell_row(ids[0], sender_id="sender", ts_ms=1_700_000_000_000)
            ]
        )
        mock_reader_cls.return_value.read = AsyncMock(return_value=[])
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = mock_aextract

        mod = importlib.import_module("everos.memory.strategies.extract_user_profile")
        monkeypatch.setattr(mod, "_reader", None, raising=False)
        monkeypatch.setattr(mod, "_writer", None, raising=False)

        await asyncio.gather(
            extract_user_profile(
                _event(owner_id=owner_a, cluster_id="cl_a"), FakeStrategyContext()
            ),
            extract_user_profile(
                _event(owner_id=owner_b, cluster_id="cl_b"), FakeStrategyContext()
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs sharing ``owner_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice", monkeypatch)
    assert log in (
        ["enter:u_alice", "leave:u_alice", "enter:u_alice", "leave:u_alice"],
    )
    # Same-owner runs always log "u_alice" twice — verify strict ordering
    # by tagging entry/leave pairs are adjacent (no interleave possible).
    assert log[0].startswith("enter:") and log[1].startswith("leave:")
    assert log[2].startswith("enter:") and log[3].startswith("leave:")


async def test_partition_lock_lets_different_owners_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs on distinct ``owner_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("u_alice", "u_bob", monkeypatch)
    assert log.index("enter:u_alice") < log.index("leave:u_bob")
    assert log.index("enter:u_bob") < log.index("leave:u_alice")
