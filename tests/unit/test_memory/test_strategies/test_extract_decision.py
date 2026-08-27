from __future__ import annotations

import datetime as _dt
import importlib
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import (
    ChatMessage,
    Decision,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

from everos.core.persistence import EntryId
from everos.infra.ome.testing import FakeStrategyContext
from everos.memory.events import DecisionExtracted, UserPipelineStarted
from everos.memory.strategies.extract_decision import extract_decision

mod = importlib.import_module("everos.memory.strategies.extract_decision")


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


def _algo_decision(
    *,
    title: str = "Use Rust on device",
    decision: str = "Device Runtime uses Rust.",
    reason: str = "Need deterministic latency.",
    impact: str | None = None,
    tags: list[str] | None = None,
) -> Decision:
    return Decision(
        owner_id=None,
        title=title,
        decision=decision,
        reason=reason,
        impact=impact,
        tags=["runtime"] if tags is None else tags,
        timestamp=1_700_000_000_000,
    )


def _event(memcell: MemCell | None = None) -> UserPipelineStarted:
    return UserPipelineStarted(
        memcell_id="mc_a",
        session_id="s1",
        memcell=memcell or _two_user_memcell(),
    )


def _eids(*seqs: int) -> list[EntryId]:
    return [EntryId(prefix="dc", date=_dt.date(2026, 5, 17), seq=seq) for seq in seqs]


async def test_strategy_meta_is_attached() -> None:
    meta = extract_decision.meta
    assert meta.name == "extract_decision"
    assert UserPipelineStarted in meta.trigger.on
    assert meta.emits == frozenset({DecisionExtracted})
    assert meta.enabled is True
    assert meta.max_retries == 2


async def test_one_llm_call_fans_out_per_user_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One aextract (no sender_id), then md + emit per user owner."""
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    algo = [
        _algo_decision(title="A", decision="choose A", reason="r1"),
        _algo_decision(title="B", decision="choose B", reason="r2"),
    ]
    event = _event()
    with (
        patch(
            "everos.memory.strategies.extract_decision.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_decision.DecisionExtractor"
        ) as mock_cls,
        patch("everos.memory.strategies.extract_decision.DecisionWriter") as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=algo)
        mock_wcls.return_value.append_entries = AsyncMock(
            side_effect=[_eids(1, 2), _eids(1, 2)]
        )
        ctx = FakeStrategyContext()
        await extract_decision(event, ctx)

    assert mock_cls.return_value.aextract.await_count == 1
    call = mock_cls.return_value.aextract.await_args
    assert call.args[0] is event.memcell
    assert "sender_id" not in call.kwargs
    assert call.kwargs.get("prompt") is None

    assert mock_wcls.return_value.append_entries.call_count == 2
    owners = [c.args[0] for c in mock_wcls.return_value.append_entries.call_args_list]
    assert owners == ["u_alice", "u_bob"]
    for c in mock_wcls.return_value.append_entries.call_args_list:
        assert len(c.args[1]) == 2
        assert c.kwargs["app_id"] == "default"
        assert c.kwargs["project_id"] == "default"

    emitted = [e for e in ctx.emitted if isinstance(e, DecisionExtracted)]
    assert len(emitted) == 4
    assert [e.owner_id for e in emitted] == [
        "u_alice",
        "u_alice",
        "u_bob",
        "u_bob",
    ]
    assert [e.decision_entry_id for e in emitted[:2]] == [
        _eids(1, 2)[0].format(),
        _eids(1, 2)[1].format(),
    ]
    assert emitted[0].title == "A"
    assert emitted[0].decision_text == "choose A"
    assert emitted[0].reason == "r1"
    assert emitted[0].source == "pipeline"
    assert emitted[0].memcell_id == "mc_a"
    assert emitted[0].session_id == "s1"
    assert emitted[0].decision_timestamp_ms == 1_700_000_000_000

    matching = [e for e in captured if e.get("event") == "decisions_extracted"]
    assert matching, "expected decisions_extracted log line"
    assert matching[0]["count"] == 4
    assert matching[0]["owner_ids"] == ["u_alice", "u_bob"]


async def test_writes_inline_and_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    algo = [
        _algo_decision(impact=None, tags=["runtime"]),
        _algo_decision(
            title="Keep local-first",
            decision="Markdown is SoT.",
            reason="Cascade projects.",
            impact="Keep Python in the agent runtime.",
            tags=[],
        ),
    ]
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
    with (
        patch(
            "everos.memory.strategies.extract_decision.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_decision.DecisionExtractor"
        ) as mock_cls,
        patch("everos.memory.strategies.extract_decision.DecisionWriter") as mock_wcls,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=algo)
        mock_wcls.return_value.append_entries = AsyncMock(return_value=_eids(1, 2))
        await extract_decision(event, FakeStrategyContext())

    assert mock_wcls.return_value.append_entries.call_count == 1
    batch = mock_wcls.return_value.append_entries.call_args
    assert batch.args[0] == "u_alice"
    items = batch.args[1]
    assert len(items) == 2

    inline0, sections0 = items[0]
    assert inline0["owner_id"] == "u_alice"
    assert inline0["session_id"] == "s1"
    assert inline0["parent_type"] == "memcell"
    assert inline0["parent_id"] == "mc_a"
    assert inline0["tags"] == ["runtime"]
    assert "sender_ids" not in inline0
    assert sections0 == {
        "Title": "Use Rust on device",
        "Decision": "Device Runtime uses Rust.",
        "Reason": "Need deterministic latency.",
    }
    assert "Impact" not in sections0

    inline1, sections1 = items[1]
    assert inline1["tags"] == []
    assert sections1["Impact"] == "Keep Python in the agent runtime."
    assert sections1["Title"] == "Keep local-first"


async def test_empty_algo_list_writes_and_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    event = _event()
    with (
        patch(
            "everos.memory.strategies.extract_decision.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_decision.DecisionExtractor"
        ) as mock_cls,
        patch("everos.memory.strategies.extract_decision.DecisionWriter") as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])
        ctx = FakeStrategyContext()
        await extract_decision(event, ctx)

    assert mock_cls.return_value.aextract.await_count == 1
    mock_wcls.return_value.append_entries.assert_not_called()
    assert ctx.emitted == []
    matching = [e for e in captured if e.get("event") == "decisions_extracted"]
    assert matching[0]["count"] == 0


async def test_skips_llm_when_memcell_has_no_user_senders(
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
            "everos.memory.strategies.extract_decision.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_decision.DecisionExtractor"
        ) as mock_cls,
        patch("everos.memory.strategies.extract_decision.DecisionWriter") as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])
        ctx = FakeStrategyContext()
        await extract_decision(event, ctx)

    mock_cls.assert_not_called()
    mock_wcls.return_value.append_entries.assert_not_called()
    matching = [e for e in captured if e.get("event") == "decisions_extracted"]
    assert matching, "log line should still fire (count=0)"
    assert matching[0]["count"] == 0


def _tool_call_memcell(*, with_user_message: bool) -> MemCell:
    items: list[object] = [
        ToolCallRequest(
            id="t1",
            sender_id="agent",
            timestamp=1_700_000_000_000,
            tool_calls=[
                ToolCall(
                    id="c1",
                    function=ToolCallFunction(name="read_file", arguments="{}"),
                )
            ],
        ),
        ToolCallResult(
            id="t2",
            timestamp=1_700_000_001_000,
            tool_call_id="c1",
            content="file contents",
        ),
    ]
    if with_user_message:
        items.insert(
            0,
            ChatMessage(
                id="m1",
                role="user",
                content="please fix the autoreloader",
                timestamp=1_699_999_999_000,
                sender_id="u_alice",
            ),
        )
    return MemCell(items=items, timestamp=1_700_000_000_000)


@pytest.mark.parametrize(
    ("with_user_message", "expect_llm"),
    [
        pytest.param(False, False, id="pure_agent_trajectory"),
        pytest.param(True, True, id="mixed_user_and_tool_calls"),
    ],
)
async def test_tool_calls_do_not_crash_the_sender_scan(
    monkeypatch: pytest.MonkeyPatch,
    with_user_message: bool,
    expect_llm: bool,
) -> None:
    event = UserPipelineStarted(
        memcell_id="mc_tool",
        session_id="s1",
        memcell=_tool_call_memcell(with_user_message=with_user_message),
    )
    monkeypatch.setattr(mod, "_writer", None, raising=False)

    with (
        patch(
            "everos.memory.strategies.extract_decision.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_decision.DecisionExtractor"
        ) as mock_cls,
        patch("everos.memory.strategies.extract_decision.DecisionWriter") as mock_wcls,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])
        await extract_decision(event, FakeStrategyContext())

    if expect_llm:
        assert mock_cls.return_value.aextract.await_count == 1
        assert "sender_id" not in mock_cls.return_value.aextract.await_args.kwargs
    else:
        mock_cls.assert_not_called()
