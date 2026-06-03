from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import AtomicFact, ChatMessage, MemCell

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory.events import UserPipelineStarted
from everos.memory.strategies.extract_atomic_facts import extract_atomic_facts

mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")


def _two_user_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi from alice",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m2",
                role="user",
                content="hi from bob",
                timestamp=1_700_000_001_000,
                sender_id="u_bob",
            ),
            ChatMessage(
                id="m3",
                role="assistant",
                content="hello both",
                timestamp=1_700_000_002_000,
                sender_id="agent",
            ),
        ],
        timestamp=1_700_000_002_000,
    )


def _fact(owner_id: str | None, text: str) -> AtomicFact:
    return AtomicFact(owner_id=owner_id, content=text, timestamp=1_700_000_000_000)


def _event() -> UserPipelineStarted:
    return UserPipelineStarted(
        memcell_id="mc_a", session_id="s1", memcell=_two_user_memcell()
    )


async def test_strategy_meta_is_attached() -> None:
    meta = extract_atomic_facts._ome_strategy_meta  # type: ignore[attr-defined]
    assert meta.name == "extract_atomic_facts"
    assert UserPipelineStarted in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 2


async def test_extracts_once_and_fans_out_per_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One LLM call per memcell; same fact list re-written under each sender.

    The algo prompt is subject-agnostic (only ``INPUT_TEXT`` + ``TIME``
    placeholders), so re-running it per sender would burn LLM tokens
    and let non-determinism drift the per-sender md files apart. The
    strategy calls ``aextract`` once with ``sender_id=None`` and
    broadcasts the resulting list — every user sender gets its own md
    entries pointing at the same fact bodies.

    Per-owner batching: the strategy collects each sender's full fact
    list and issues one :meth:`append_entries` per owner (not N single
    appends), so the call shape is one batch call per sender.
    """
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    generic_facts = [
        _fact(None, "alice mentioned a weekend trip to tokyo"),
        _fact(None, "bob said he needs hiking gear"),
    ]

    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=generic_facts)
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])

        await extract_atomic_facts(_event(), FakeStrategyContext())

    # Exactly one LLM call, parameterised with sender_id=None.
    assert mock_cls.return_value.aextract.await_count == 1
    call = mock_cls.return_value.aextract.call_args
    assert call.kwargs["sender_id"] is None

    # 2 senders → 2 batch calls; each batch carries this sender's 2 facts
    # (same generic body re-used).
    assert mock_wcls.return_value.append_entries.call_count == 2
    batch_calls = mock_wcls.return_value.append_entries.call_args_list
    batched_owners = sorted(c.args[0] for c in batch_calls)
    assert batched_owners == ["u_alice", "u_bob"]
    # Flatten items across batches: (owner, fact_text) pairs.
    flat = sorted(
        (c.args[0], sections["Fact"])
        for c in batch_calls
        for inline, sections in c.args[1]
    )
    assert flat == [
        ("u_alice", "alice mentioned a weekend trip to tokyo"),
        ("u_alice", "bob said he needs hiking gear"),
        ("u_bob", "alice mentioned a weekend trip to tokyo"),
        ("u_bob", "bob said he needs hiking gear"),
    ]

    matching = [e for e in captured if e.get("event") == "atomic_facts_extracted"]
    assert matching, "expected atomic_facts_extracted log line"
    record = matching[0]
    assert record["count"] == 4
    assert sorted(record["owner_ids"]) == ["u_alice", "u_bob"]


async def test_writes_md_for_each_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = [
        _fact("u_alice", "alice likes hiking"),
        _fact("u_alice", "alice lives in tokyo"),
    ]

    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactWriter"
        ) as mock_wcls,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=facts)
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])

        event = UserPipelineStarted(
            memcell_id="mc_a",
            session_id="s1",
            memcell=MemCell(
                items=[
                    ChatMessage(
                        id="m1",
                        role="user",
                        content="hi",
                        timestamp=1_700_000_000_000,
                        sender_id="u_alice",
                    )
                ],
                timestamp=1_700_000_000_000,
            ),
        )
        await extract_atomic_facts(event, FakeStrategyContext())

    # Single sender (u_alice) → one batch call with 2 items.
    assert mock_wcls.return_value.append_entries.call_count == 1
    batch_call = mock_wcls.return_value.append_entries.call_args
    assert batch_call.args[0] == "u_alice"
    items = batch_call.args[1]
    assert len(items) == 2
    for (inline, sections), fact in zip(items, facts, strict=True):
        assert inline["owner_id"] == "u_alice"
        assert inline["session_id"] == "s1"
        assert inline["parent_type"] == "memcell"
        assert inline["parent_id"] == "mc_a"
        assert "sender_ids" not in inline
        assert sections == {"Fact": fact.content}


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
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entries = AsyncMock(return_value=[])
        ctx = FakeStrategyContext()
        await extract_atomic_facts(event, ctx)

    matching = [e for e in captured if e.get("event") == "atomic_facts_extracted"]
    assert matching, "log line should still fire (count=0)"
    assert matching[0]["count"] == 0
    mock_wcls.return_value.append_entries.assert_not_called()
