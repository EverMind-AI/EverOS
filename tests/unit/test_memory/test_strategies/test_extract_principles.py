"""Tests for :func:`extract_principles`.

Heavy mocking — the strategy threads through ``cluster_repo`` (sqlite
``kind=decision``), ``DecisionReader`` (md SoT), ``decision_repo``
(Lance fallback), ``ProfileWriter`` (principles.md), and
``PrincipleExtractor`` (algo). We mock all seams so the test exercises
the orchestration only: union every cluster, snapshot the triggering
row, persist once.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import Principle as AlgoPrinciple

from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import PrincipleFrontmatter
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import DecisionClusterUpdated
from everos.memory.strategies.extract_principles import extract_principles
from everos.service.memorize import _STRATEGIES_ALWAYS, _STRATEGIES_REQUIRE_EMBED

_MOD = "everos.memory.strategies.extract_principles"


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    cluster_id: str = "cl_dec000000001",
    decision_entry_id: str = "dc_20260101_0001",
    title: str = "Use Rust runtime",
    decision_text: str = "Device Runtime is implemented in Rust.",
    reason: str = "Need deterministic latency.",
    impact: str | None = None,
    tags: list[str] | None = None,
    decision_timestamp_ms: int = 1_700_000_001_000,
) -> DecisionClusterUpdated:
    return DecisionClusterUpdated(
        memcell_id=memcell_id,
        decision_entry_id=decision_entry_id,
        cluster_id=cluster_id,
        owner_id=owner_id,
        title=title,
        decision_text=decision_text,
        reason=reason,
        impact=impact,
        tags=list(tags or ["runtime"]),
        decision_timestamp_ms=decision_timestamp_ms,
    )


def _algo_cluster(*, cluster_id: str, members: list[str]) -> AlgoCluster:
    return AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(1024, dtype=np.float32),
        count=len(members),
        last_ts=1_700_000_001_000,
        preview=[],
        members=members,
    )


def _principle(
    *, title: str, statement: str, source_entry_ids: list[str]
) -> AlgoPrinciple:
    return AlgoPrinciple(
        owner_id="u_alice",
        title=title,
        statement=statement,
        source_entry_ids=source_entry_ids,
        timestamp=1_700_000_001_000,
    )


def _structured(
    *,
    title: str,
    decision: str,
    reason: str = "why",
    tags: str = "[runtime]",
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> SimpleNamespace:
    return SimpleNamespace(
        sections={"Title": title, "Decision": decision, "Reason": reason},
        inline={"tags": tags, "timestamp": timestamp},
    )


async def test_strategy_meta_is_attached() -> None:
    meta = extract_principles.meta
    assert meta.name == "extract_principles"
    assert DecisionClusterUpdated in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 2
    assert extract_principles in _STRATEGIES_ALWAYS
    assert extract_principles not in _STRATEGIES_REQUIRE_EMBED


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    event: DecisionClusterUpdated,
    *,
    clusters: list[AlgoCluster],
    extractor: AsyncMock,
    structured_by_id: dict[str, SimpleNamespace] | None = None,
    lance_by_id: dict[str, object] | None = None,
    mint_ids: list[str] | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Run extract_principles under the standard mock stack."""
    with (
        patch(f"{_MOD}.cluster_repo") as mock_cluster_repo,
        patch(f"{_MOD}.DecisionReader") as mock_reader_cls,
        patch(f"{_MOD}.decision_repo") as mock_decision_repo,
        patch(f"{_MOD}.get_llm_client", return_value=object()),
        patch(f"{_MOD}.PrincipleExtractor") as mock_extractor_cls,
        patch(f"{_MOD}.ProfileWriter") as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=clusters)
        structured = structured_by_id or {}

        async def _find_structured(_owner: str, entry_id: str, **_kw: object):
            return structured.get(entry_id)

        mock_reader_cls.return_value.find_structured = AsyncMock(
            side_effect=_find_structured
        )
        lance = lance_by_id or {}

        async def _find_lance(_owner: str, entry_id: str, **_kw: object):
            return lance.get(entry_id)

        mock_decision_repo.find_by_owner_entry = AsyncMock(side_effect=_find_lance)
        mock_extractor_cls.return_value.aextract = extractor
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)

        mod = importlib.import_module(_MOD)
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_decision_reader", None, raising=False)
        if mint_ids is not None:
            ids = iter(mint_ids)
            monkeypatch.setattr(mod, "mint_principle_id", lambda: next(ids))

        await extract_principles(event, FakeStrategyContext())
        write = mock_writer_cls.return_value.write
        aextract = mock_extractor_cls.return_value.aextract
        return write, aextract


async def test_unions_all_decision_clusters_into_one_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two clusters → extractor twice (once each); writer once with the union.

    Extracting only the triggering cluster and rewriting principles.md
    would wipe cluster B — that is the would-wipe regression this pins.
    """
    cluster_a = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    cluster_b = _algo_cluster(cluster_id="cl_b", members=["dc_20260101_0002"])
    p_a = _principle(
        title="From A",
        statement="Prefer the snapshot cluster.",
        source_entry_ids=["dc_20260101_0001"],
    )
    p_b = _principle(
        title="From B",
        statement="Keep the other cluster.",
        source_entry_ids=["dc_20260101_0002"],
    )

    async def fake_extract(decisions, *, owner_id, **_kw):
        eid = decisions[0][0]
        if eid == "dc_20260101_0001":
            return [p_a]
        return [p_b]

    write, aextract = await _run(
        monkeypatch,
        _event(),
        clusters=[cluster_a, cluster_b],
        extractor=AsyncMock(side_effect=fake_extract),
        structured_by_id={
            "dc_20260101_0002": _structured(title="Other", decision="Keep cluster B.")
        },
        mint_ids=["pr_aaaaaaaaaaaa", "pr_bbbbbbbbbbbb"],
    )

    assert aextract.await_count == 2
    write.assert_awaited_once()
    kwargs = write.await_args.kwargs
    fm = kwargs["frontmatter"]
    assert isinstance(fm, PrincipleFrontmatter)
    assert fm.id == "principle_u_alice"
    assert fm.user_id == "u_alice"
    titles = [item.title for item in fm.principles]
    assert titles == ["From A", "From B"]
    assert [item.id for item in fm.principles] == [
        "pr_aaaaaaaaaaaa",
        "pr_bbbbbbbbbbbb",
    ]
    assert "Keep the other cluster." in kwargs["body"]


async def test_list_for_owner_uses_kind_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    with (
        patch(f"{_MOD}.cluster_repo") as mock_cluster_repo,
        patch(f"{_MOD}.DecisionReader") as mock_reader_cls,
        patch(f"{_MOD}.decision_repo") as mock_decision_repo,
        patch(f"{_MOD}.get_llm_client", return_value=object()),
        patch(f"{_MOD}.PrincipleExtractor") as mock_extractor_cls,
        patch(f"{_MOD}.ProfileWriter") as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(return_value=[cluster])
        mock_reader_cls.return_value.find_structured = AsyncMock(return_value=None)
        mock_decision_repo.find_by_owner_entry = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mod = importlib.import_module(_MOD)
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_decision_reader", None, raising=False)
        await extract_principles(_event(), FakeStrategyContext())

    args, kwargs = mock_cluster_repo.list_for_owner.await_args
    assert args[0] == "u_alice"
    assert args[1] == "decision"
    assert kwargs["app_id"] == "default"
    assert kwargs["project_id"] == "default"


async def test_triggering_member_uses_event_snapshot_when_reader_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    captured: list[list] = []

    async def fake_extract(decisions, *, owner_id, **_kw):
        captured.append(list(decisions))
        return []

    await _run(
        monkeypatch,
        _event(decision_text="Device Runtime is implemented in Rust."),
        clusters=[cluster],
        extractor=AsyncMock(side_effect=fake_extract),
        structured_by_id={},
        lance_by_id={},
    )

    assert len(captured) == 1
    eid, decision = captured[0][0]
    assert eid == "dc_20260101_0001"
    assert decision.decision == "Device Runtime is implemented in Rust."
    assert decision.title == "Use Rust runtime"
    assert decision.tags == ["runtime"]


async def test_other_members_load_via_decision_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _algo_cluster(
        cluster_id="cl_a",
        members=["dc_20260101_0001", "dc_20260101_0002"],
    )
    captured: list[list] = []

    async def fake_extract(decisions, *, owner_id, **_kw):
        captured.append(list(decisions))
        return []

    await _run(
        monkeypatch,
        _event(),
        clusters=[cluster],
        extractor=AsyncMock(side_effect=fake_extract),
        structured_by_id={
            "dc_20260101_0002": _structured(
                title="Keep Go",
                decision="Leave the control plane in Go.",
            )
        },
    )

    by_id = {eid: dc for eid, dc in captured[0]}
    assert by_id["dc_20260101_0001"].decision.startswith("Device Runtime")
    assert by_id["dc_20260101_0002"].decision == "Leave the control plane in Go."
    assert by_id["dc_20260101_0002"].title == "Keep Go"


async def test_empty_extractor_list_still_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    write, aextract = await _run(
        monkeypatch,
        _event(),
        clusters=[cluster],
        extractor=AsyncMock(return_value=[]),
    )
    aextract.assert_awaited_once()
    write.assert_awaited_once()
    fm = write.await_args.kwargs["frontmatter"]
    assert fm.principles == []


async def test_no_clusters_writes_empty_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write, aextract = await _run(
        monkeypatch,
        _event(),
        clusters=[],
        extractor=AsyncMock(
            return_value=[
                _principle(
                    title="stale",
                    statement="should not run",
                    source_entry_ids=["x"],
                )
            ]
        ),
    )
    aextract.assert_not_awaited()
    write.assert_awaited_once()
    assert write.await_args.kwargs["frontmatter"].principles == []


async def test_unloadable_members_skip_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Members exist but none load (no snapshot text, no md, no lance)."""
    cluster = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    write, aextract = await _run(
        monkeypatch,
        _event(decision_text=""),
        clusters=[cluster],
        extractor=AsyncMock(return_value=[]),
    )
    aextract.assert_not_awaited()
    write.assert_not_awaited()


async def test_does_not_write_if_a_cluster_extract_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster_a = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    cluster_b = _algo_cluster(cluster_id="cl_b", members=["dc_20260101_0002"])
    with (
        patch(f"{_MOD}.cluster_repo") as mock_cluster_repo,
        patch(f"{_MOD}.DecisionReader") as mock_reader_cls,
        patch(f"{_MOD}.decision_repo") as mock_decision_repo,
        patch(f"{_MOD}.get_llm_client", return_value=object()),
        patch(f"{_MOD}.PrincipleExtractor") as mock_extractor_cls,
        patch(f"{_MOD}.ProfileWriter") as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            return_value=[cluster_a, cluster_b]
        )
        mock_reader_cls.return_value.find_structured = AsyncMock(
            return_value=_structured(title="B", decision="Keep B.")
        )
        mock_decision_repo.find_by_owner_entry = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(
            side_effect=[
                [
                    _principle(
                        title="A",
                        statement="a",
                        source_entry_ids=["dc_20260101_0001"],
                    )
                ],
                RuntimeError("llm down"),
            ]
        )
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mod = importlib.import_module(_MOD)
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_decision_reader", None, raising=False)
        with pytest.raises(RuntimeError, match="llm down"):
            await extract_principles(_event(), FakeStrategyContext())
    mock_writer_cls.return_value.write.assert_not_awaited()


async def _run_serialisation_probe(
    owner_a: str, owner_b: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    log: list[str] = []

    async def mock_aextract(decisions, *, owner_id, **_kwargs):
        log.append(f"enter:{owner_id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{owner_id}")
        return []

    cluster_a = _algo_cluster(cluster_id="cl_a", members=["dc_20260101_0001"])
    cluster_b = _algo_cluster(cluster_id="cl_b", members=["dc_20260101_0002"])

    with (
        patch(f"{_MOD}.cluster_repo") as mock_cluster_repo,
        patch(f"{_MOD}.DecisionReader") as mock_reader_cls,
        patch(f"{_MOD}.decision_repo") as mock_decision_repo,
        patch(f"{_MOD}.get_llm_client", return_value=object()),
        patch(f"{_MOD}.PrincipleExtractor") as mock_extractor_cls,
        patch(f"{_MOD}.ProfileWriter") as mock_writer_cls,
    ):
        mock_cluster_repo.list_for_owner = AsyncMock(
            side_effect=lambda owner, _kind, **_kw: (
                [cluster_a] if owner == owner_a else [cluster_b]
            )
        )
        mock_reader_cls.return_value.find_structured = AsyncMock(return_value=None)
        mock_decision_repo.find_by_owner_entry = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = mock_aextract
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)

        mod = importlib.import_module(_MOD)
        monkeypatch.setattr(mod, "_writer", None, raising=False)
        monkeypatch.setattr(mod, "_decision_reader", None, raising=False)

        await asyncio.gather(
            extract_principles(
                _event(
                    owner_id=owner_a,
                    cluster_id="cl_a",
                    decision_entry_id="dc_20260101_0001",
                    decision_text="A",
                ),
                FakeStrategyContext(),
            ),
            extract_principles(
                _event(
                    owner_id=owner_b,
                    cluster_id="cl_a" if owner_b == owner_a else "cl_b",
                    decision_entry_id=(
                        "dc_20260101_0001"
                        if owner_b == owner_a
                        else "dc_20260101_0002"
                    ),
                    decision_text="B",
                ),
                FakeStrategyContext(),
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = await _run_serialisation_probe("u_alice", "u_alice", monkeypatch)
    assert log[0].startswith("enter:") and log[1].startswith("leave:")
    assert log[2].startswith("enter:") and log[3].startswith("leave:")


async def test_partition_lock_lets_different_owners_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = await _run_serialisation_probe("u_alice", "u_bob", monkeypatch)
    assert sorted(log) == sorted(
        ["enter:u_alice", "leave:u_alice", "enter:u_bob", "leave:u_bob"]
    )
