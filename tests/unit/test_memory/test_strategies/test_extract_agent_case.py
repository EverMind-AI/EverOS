from __future__ import annotations

import datetime as _dt
import importlib
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing
from everalgo.types import AgentCase, ChatMessage, MemCell

from everos.core.persistence import EntryId
from everos.infra.ome.testing import FakeStrategyContext
from everos.memory.events import AgentCaseExtracted, AgentPipelineStarted
from everos.memory.strategies.extract_agent_case import extract_agent_case


def _fake_eid() -> EntryId:
    return EntryId(prefix="ac", date=_dt.date(2026, 5, 17), seq=1)


mod = importlib.import_module("everos.memory.strategies.extract_agent_case")


def _agent_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="please summarise the doc",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="here's the summary ...",
                timestamp=1_700_000_001_000,
                sender_id="agent_42",
            ),
        ],
        timestamp=1_700_000_001_000,
    )


def _event() -> AgentPipelineStarted:
    return AgentPipelineStarted(
        memcell_id="mc_a", session_id="s1", memcell=_agent_memcell()
    )


def _algo_case(
    *,
    task_intent: str = "summarise doc",
    approach: str = "read + condense",
    quality_score: float = 0.8,
    key_insight: str = "",
) -> AgentCase:
    return AgentCase(
        id=uuid.uuid4().hex,
        timestamp=1_700_000_001_000,
        task_intent=task_intent,
        approach=approach,
        quality_score=quality_score,
        key_insight=key_insight,
    )


async def test_strategy_meta_is_attached() -> None:
    meta = extract_agent_case._ome_strategy_meta  # type: ignore[attr-defined]
    assert meta.name == "extract_agent_case"
    assert AgentPipelineStarted in meta.trigger.on
    assert meta.emits == frozenset({AgentCaseExtracted})
    assert meta.max_retries == 2


async def test_writes_md_when_algo_returns_a_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    case = _algo_case(quality_score=0.9, key_insight="batch-then-summarise")
    with (
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[case])
        mock_wcls.return_value.append_entry = AsyncMock(return_value=_fake_eid())
        ctx = FakeStrategyContext()

        await extract_agent_case(_event(), ctx)

    assert mock_cls.return_value.aextract.await_count == 1
    assert mock_wcls.return_value.append_entry.call_count == 1
    _, kwargs = mock_wcls.return_value.append_entry.call_args
    assert kwargs["inline"]["owner_id"] == "agent_42"
    assert kwargs["inline"]["session_id"] == "s1"
    assert kwargs["inline"]["parent_type"] == "memcell"
    assert kwargs["inline"]["parent_id"] == "mc_a"
    assert kwargs["inline"]["quality_score"] == 0.9
    assert kwargs["sections"] == {
        "TaskIntent": "summarise doc",
        "Approach": "read + condense",
        "KeyInsight": "batch-then-summarise",
    }
    # Chain emit: AgentCaseExtracted fires after the md write.
    emitted = [e for e in ctx.emitted if isinstance(e, AgentCaseExtracted)]
    assert len(emitted) == 1
    assert emitted[0].memcell_id == "mc_a"
    assert emitted[0].case_entry_id == _fake_eid().format()
    assert emitted[0].task_intent == "summarise doc"
    assert emitted[0].quality_score == 0.9
    assert emitted[0].case_timestamp_ms == 1_700_000_001_000
    assert emitted[0].agent_id == "agent_42"

    matching = [e for e in captured if e.get("event") == "agent_case_extracted"]
    assert matching, "expected agent_case_extracted log line"
    assert matching[0]["owner_ids"] == ["agent_42"]
    assert matching[0]["fanout"] == 1
    assert matching[0]["quality_score"] == 0.9


async def test_fans_out_per_assistant_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One LLM call, then md write + emit per distinct assistant sender.

    Case text is third-person (``the agent did X``) so the same body
    is a valid reference experience for every assistant sender that
    participated in the trajectory. Verifies: aextract is called
    exactly once, md is written once per agent, and an
    ``AgentCaseExtracted`` event fires per agent so the downstream
    skill clustering chain runs in each agent's own scope.
    """
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    multi_agent_cell = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="please dispatch",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="dispatching to specialist",
                timestamp=1_700_000_001_000,
                sender_id="agent_lead",
            ),
            ChatMessage(
                id="m3",
                role="assistant",
                content="here is the answer",
                timestamp=1_700_000_002_000,
                sender_id="agent_specialist",
            ),
        ],
        timestamp=1_700_000_002_000,
    )
    event = AgentPipelineStarted(
        memcell_id="mc_multi", session_id="s_multi", memcell=multi_agent_cell
    )
    case = _algo_case(quality_score=0.85)

    # writer.append_entry returns a different entry_id per call so the
    # emitted events carry per-agent entry_ids (cascade keys off owner+entry).
    eids = [
        EntryId(prefix="ac", date=_dt.date(2026, 5, 17), seq=1),
        EntryId(prefix="ac", date=_dt.date(2026, 5, 17), seq=2),
    ]

    with (
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[case])
        mock_wcls.return_value.append_entry = AsyncMock(side_effect=eids)
        ctx = FakeStrategyContext()

        await extract_agent_case(event, ctx)

    # Exactly one LLM call regardless of agent count.
    assert mock_cls.return_value.aextract.await_count == 1

    # Two md writes (one per distinct assistant sender), in first-seen order.
    assert mock_wcls.return_value.append_entry.call_count == 2
    owners_written = [
        call.kwargs["inline"]["owner_id"]
        for call in mock_wcls.return_value.append_entry.call_args_list
    ]
    assert owners_written == ["agent_lead", "agent_specialist"]

    # Two emits, each tagged with its own agent_id + per-agent entry_id.
    emitted = [e for e in ctx.emitted if isinstance(e, AgentCaseExtracted)]
    assert len(emitted) == 2
    assert [e.agent_id for e in emitted] == ["agent_lead", "agent_specialist"]
    assert [e.case_entry_id for e in emitted] == [eids[0].format(), eids[1].format()]
    # Same task body / quality across the fan-out (broadcast semantics).
    assert {e.task_intent for e in emitted} == {"summarise doc"}
    assert {e.quality_score for e in emitted} == {0.85}

    matching = [e for e in captured if e.get("event") == "agent_case_extracted"]
    assert matching, "expected agent_case_extracted log line"
    assert matching[0]["owner_ids"] == ["agent_lead", "agent_specialist"]
    assert matching[0]["fanout"] == 2


async def test_omits_key_insight_section_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    case = _algo_case(key_insight="")
    with (
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseWriter"
        ) as mock_wcls,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[case])
        mock_wcls.return_value.append_entry = AsyncMock(return_value=_fake_eid())
        await extract_agent_case(_event(), FakeStrategyContext())

    _, kwargs = mock_wcls.return_value.append_entry.call_args
    assert "KeyInsight" not in kwargs["sections"]
    assert kwargs["sections"]["TaskIntent"] == "summarise doc"


async def test_skips_when_algo_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Algo pre-filter rejected the cell — no md written, log a noop line."""
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entry = AsyncMock(return_value=_fake_eid())
        await extract_agent_case(_event(), FakeStrategyContext())

    mock_wcls.return_value.append_entry.assert_not_called()
    matching = [e for e in captured if e.get("event") == "agent_case_skipped_by_algo"]
    assert matching, "expected agent_case_skipped_by_algo log line"


async def test_skips_when_no_assistant_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No assistant in the cell → no agent_id can be inferred; algo not called."""
    user_only = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
        ],
        timestamp=1_700_000_000_000,
    )
    event = AgentPipelineStarted(memcell_id="mc_b", session_id="s1", memcell=user_only)

    monkeypatch.setattr(mod, "_writer", None, raising=False)
    with (
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_cls,
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseWriter"
        ) as mock_wcls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        mock_wcls.return_value.append_entry = AsyncMock(return_value=_fake_eid())
        await extract_agent_case(event, FakeStrategyContext())

    # Algo extractor must not be invoked at all when there's no agent.
    mock_cls.return_value.aextract.assert_not_called()
    mock_wcls.return_value.append_entry.assert_not_called()
    matching = [
        e for e in captured if e.get("event") == "agent_case_skipped_no_assistant"
    ]
    assert matching, "expected agent_case_skipped_no_assistant log line"
