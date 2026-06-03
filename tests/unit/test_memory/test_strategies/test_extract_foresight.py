from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import ChatMessage, Foresight, MemCell

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory.events import UserPipelineStarted
from everos.memory.strategies.extract_foresight import extract_foresight

mod = importlib.import_module("everos.memory.strategies.extract_foresight")


def _two_user_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="alice plans a trip",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m2",
                role="user",
                content="bob will buy tickets",
                timestamp=1_700_000_001_000,
                sender_id="u_bob",
            ),
            ChatMessage(
                id="m3",
                role="assistant",
                content="sounds good",
                timestamp=1_700_000_002_000,
                sender_id="agent",
            ),
        ],
        timestamp=1_700_000_002_000,
    )


def _foresight(owner_id: str, text: str) -> Foresight:
    return Foresight(
        owner_id=owner_id,
        foresight=text,
        evidence="...",
        timestamp=1_700_000_000_000,
    )


def _event() -> UserPipelineStarted:
    return UserPipelineStarted(
        memcell_id="mc_a", session_id="s1", memcell=_two_user_memcell()
    )


async def test_strategy_meta_is_attached() -> None:
    meta = extract_foresight._ome_strategy_meta  # type: ignore[attr-defined]
    assert meta.name == "extract_foresight"
    assert UserPipelineStarted in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 2


async def test_extracts_per_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-sender extraction (like Episode, unlike AtomicFact's fan-out)."""
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_foresight.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_foresight.ForesightExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_foresight.ForesightWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        # sender_ids in the strategy are sorted: alice first, bob second.
        mock_cls.return_value.aextract = AsyncMock(
            side_effect=[
                [_foresight("u_alice", "trip to tokyo")],
                [_foresight("u_bob", "buy plane tickets")],
            ]
        )
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])

        await extract_foresight(_event(), FakeStrategyContext())

    # Per-sender semantics: one LLM call per user sender.
    assert mock_cls.return_value.aextract.await_count == 2
    sender_id_calls = [
        call.kwargs.get("sender_id")
        for call in mock_cls.return_value.aextract.call_args_list
    ]
    assert sender_id_calls == ["u_alice", "u_bob"]

    # Per-owner batching: one batch call per owner; here each owner has 1
    # foresight, so two batches each carrying 1 item.
    assert mock_wcls.return_value.append_entries.call_count == 2
    batched_owners = sorted(
        c.args[0] for c in mock_wcls.return_value.append_entries.call_args_list
    )
    assert batched_owners == ["u_alice", "u_bob"]

    matching = [e for e in captured if e.get("event") == "foresights_extracted"]
    assert matching, "expected foresights_extracted log line"
    record = matching[0]
    assert record["count"] == 2
    assert sorted(record["owner_ids"]) == ["u_alice", "u_bob"]


async def test_writes_md_for_each_foresight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foresights = [
        Foresight(
            owner_id="u_alice",
            foresight="trip to tokyo",
            evidence="said so",
            timestamp=1_700_000_000_000,
        ),
        Foresight(
            owner_id="u_alice",
            foresight="buy tickets",
            evidence="confirmed",
            timestamp=1_700_000_000_000,
            start_time="2023-11-15",
            duration_days=7,
        ),
    ]

    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_foresight.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_foresight.ForesightExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_foresight.ForesightWriter"
        ) as mock_wcls,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=foresights)
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])

        event = UserPipelineStarted(
            memcell_id="mc_a",
            session_id="s1",
            memcell=MemCell(
                items=[
                    ChatMessage(
                        id="m1",
                        role="user",
                        content="planning a trip",
                        timestamp=1_700_000_000_000,
                        sender_id="u_alice",
                    )
                ],
                timestamp=1_700_000_000_000,
            ),
        )
        await extract_foresight(event, FakeStrategyContext())

    # Single sender (u_alice) → one batch call with both foresights.
    assert mock_wcls.return_value.append_entries.call_count == 1
    batch_call = mock_wcls.return_value.append_entries.call_args
    assert batch_call.args[0] == "u_alice"
    items = batch_call.args[1]
    assert len(items) == 2

    # First foresight: no optional time fields
    inline0, sections0 = items[0]
    assert inline0["owner_id"] == "u_alice"
    assert inline0["session_id"] == "s1"
    assert inline0["parent_type"] == "memcell"
    assert inline0["parent_id"] == "mc_a"
    assert "sender_ids" not in inline0
    assert "start_time" not in inline0
    assert "duration_days" not in inline0
    assert sections0 == {"Foresight": "trip to tokyo", "Evidence": "said so"}

    # Second foresight: has start_time + duration_days
    inline1, sections1 = items[1]
    assert inline1["start_time"] == "2023-11-15"
    assert inline1["duration_days"] == 7
    assert sections1 == {"Foresight": "buy tickets", "Evidence": "confirmed"}


async def test_skips_when_memcell_has_no_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = UserPipelineStarted(
        memcell_id="mc_b",
        session_id="s1",
        memcell=MemCell(items=[], timestamp=1_700_000_000_000),
    )

    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_foresight.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_foresight.ForesightExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_foresight.ForesightWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])
        ctx = FakeStrategyContext()
        await extract_foresight(event, ctx)

    matching = [e for e in captured if e.get("event") == "foresights_extracted"]
    assert matching, "log line should still fire (count=0)"
    assert matching[0]["count"] == 0
    mock_wcls.return_value.append_entries.assert_not_called()
