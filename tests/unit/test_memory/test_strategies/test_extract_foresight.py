from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import (
    ChatMessage,
    Foresight,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

from everos.memory.events import UserPipelineStarted
from everos.memory.strategies import EXTRACT_FORESIGHT_SPEC, extract_foresight
from tests.fakes import make_deps

_EXTRACTOR = "everos.memory.strategies.extract_foresight.ForesightExtractor"


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


def test_spec() -> None:
    spec = EXTRACT_FORESIGHT_SPEC
    assert spec.name == "extract_foresight"
    assert UserPipelineStarted in spec.trigger.on
    assert spec.emits == frozenset()
    assert spec.max_retries == 2
    assert spec.enabled is False


async def test_extracts_per_sender() -> None:
    """Per-sender extraction (like Episode, unlike AtomicFact's fan-out)."""
    deps = make_deps()
    with (
        patch(_EXTRACTOR) as mock_cls,
        structlog.testing.capture_logs() as captured,
    ):
        # sender_ids in the strategy are sorted: alice first, bob second.
        mock_cls.return_value.aextract = AsyncMock(
            side_effect=[
                [_foresight("u_alice", "trip to tokyo")],
                [_foresight("u_bob", "buy plane tickets")],
            ]
        )
        events = await extract_foresight(_event(), deps)

    assert events == []
    # Per-sender semantics: one LLM call per user sender.
    sender_id_calls = [
        call.kwargs.get("sender_id")
        for call in mock_cls.return_value.aextract.call_args_list
    ]
    assert sender_id_calls == ["u_alice", "u_bob"]

    # Each foresight lands under its own owner.
    stored = deps.stores.foresight.records
    assert sorted((s.scope.owner_id, s.item.foresight) for s in stored) == [
        ("u_alice", "trip to tokyo"),
        ("u_bob", "buy plane tickets"),
    ]

    record = next(e for e in captured if e.get("event") == "foresights_extracted")
    assert record["count"] == 2
    assert sorted(record["owner_ids"]) == ["u_alice", "u_bob"]


async def test_writes_every_foresight_of_a_sender_in_one_batch() -> None:
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
    deps = make_deps()
    with patch(_EXTRACTOR) as mock_cls:
        mock_cls.return_value.aextract = AsyncMock(return_value=foresights)
        await extract_foresight(event, deps)

    stored = deps.stores.foresight.records
    assert [s.scope.owner_id for s in stored] == ["u_alice", "u_alice"]
    first, second = (s.item for s in stored)
    assert (first.session_id, first.parent_id) == ("s1", "mc_a")
    assert (first.foresight, first.evidence) == ("trip to tokyo", "said so")
    assert first.start_time is None
    assert first.duration_days is None
    assert (second.start_time, second.duration_days) == ("2023-11-15", 7)
    assert (second.foresight, second.evidence) == ("buy tickets", "confirmed")


async def test_skips_when_memcell_has_no_messages() -> None:
    event = UserPipelineStarted(
        memcell_id="mc_b",
        session_id="s1",
        memcell=MemCell(items=[], timestamp=1_700_000_000_000),
    )
    deps = make_deps()
    with (
        patch(_EXTRACTOR) as mock_cls,
        structlog.testing.capture_logs() as captured,
    ):
        await extract_foresight(event, deps)

    record = next(e for e in captured if e.get("event") == "foresights_extracted")
    assert record["count"] == 0
    mock_cls.assert_not_called()
    assert deps.stores.foresight.records == []


# ── mixed agent/user memcells (tool calls) ──────────────────────────────


def _tool_call_memcell(*, with_user_message: bool) -> MemCell:
    """A memcell shaped the way an agent trajectory arrives.

    ``ToolCallRequest`` carries ``sender_id`` but no ``role``;
    ``ToolCallResult`` carries neither. Only ``ChatMessage`` has ``role``,
    which is why a bare ``m.role`` test raised on the first tool call.
    """
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
    ("with_user_message", "expected_senders"),
    [
        pytest.param(False, [], id="pure_agent_trajectory"),
        pytest.param(True, ["u_alice"], id="mixed_user_and_tool_calls"),
    ],
)
async def test_tool_calls_do_not_crash_the_sender_scan(
    with_user_message: bool,
    expected_senders: list[str],
) -> None:
    """A memcell holding tool calls must not raise, and must not over-extract.

    Regression guard: the scan used to read ``m.role`` off every item, so
    the first ``ToolCallRequest`` raised ``AttributeError`` — before any
    sender was resolved, before any LLM call. That made the strategy sound
    on plain user chat and guaranteed to dead-letter on agent
    trajectories, which is the shape ``/add`` receives from an agent
    session. everalgo contracts for the mixed case
    (``user_memory/_render.chat_messages``), so the fix is to honour that
    contract rather than pre-filter by hand.

    Both directions are pinned: a pure agent trajectory extracts nothing
    and never reaches the LLM, and a mixed memcell extracts for the human
    senders only — an implementation that merely stopped raising but
    scanned tool-call ``sender_id`` values would invent ``"agent"`` as a
    user.
    """
    event = UserPipelineStarted(
        memcell_id="mc_tool",
        session_id="s1",
        memcell=_tool_call_memcell(with_user_message=with_user_message),
    )
    with patch(_EXTRACTOR) as mock_cls:
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        await extract_foresight(event, make_deps())

    called_senders = [
        call.kwargs["sender_id"]
        for call in mock_cls.return_value.aextract.await_args_list
    ]
    assert called_senders == expected_senders
