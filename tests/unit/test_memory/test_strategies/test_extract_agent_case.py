from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import structlog.testing
from everalgo.types import AgentCase, ChatMessage, MemCell

from everos.memory.events import AgentCaseExtracted, AgentPipelineStarted
from everos.memory.strategies import EXTRACT_AGENT_CASE_SPEC, extract_agent_case
from tests.fakes import make_deps

_EXTRACTOR = "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"


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


def test_spec() -> None:
    spec = EXTRACT_AGENT_CASE_SPEC
    assert spec.name == "extract_agent_case"
    assert AgentPipelineStarted in spec.trigger.on
    assert spec.emits == frozenset({AgentCaseExtracted})
    assert spec.max_retries == 2


async def test_writes_the_case_and_announces_it() -> None:
    deps = make_deps()
    case = _algo_case(quality_score=0.9, key_insight="batch-then-summarise")
    with (
        patch(_EXTRACTOR) as mock_cls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[case])
        events = await extract_agent_case(_event(), deps)

    assert mock_cls.return_value.aextract.await_count == 1
    (stored,) = deps.stores.agent_case.records
    assert (stored.scope.owner_id, stored.scope.owner_type) == ("agent_42", "agent")
    assert (stored.item.session_id, stored.item.parent_id) == ("s1", "mc_a")
    assert stored.item.quality_score == 0.9
    assert (stored.item.task_intent, stored.item.approach) == (
        "summarise doc",
        "read + condense",
    )
    assert stored.item.key_insight == "batch-then-summarise"

    (announced,) = events
    assert isinstance(announced, AgentCaseExtracted)
    assert announced.memcell_id == "mc_a"
    assert announced.case_entry_id == stored.ref.entry_id
    assert announced.task_intent == "summarise doc"
    assert announced.quality_score == 0.9
    assert announced.case_timestamp_ms == 1_700_000_001_000
    assert announced.agent_id == "agent_42"

    record = next(e for e in captured if e.get("event") == "agent_case_extracted")
    assert record["owner_ids"] == ["agent_42"]
    assert record["fanout"] == 1
    assert record["quality_score"] == 0.9


async def test_fans_out_per_assistant_sender() -> None:
    """One LLM call, then a write + an event per distinct assistant sender.

    Case text is third-person (``the agent did X``) so the same body
    is a valid reference experience for every assistant sender that
    participated in the trajectory. Verifies: aextract is called
    exactly once, the case is written once per agent, and an
    ``AgentCaseExtracted`` event is returned per agent so the downstream
    skill clustering chain runs in each agent's own scope.
    """
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
    deps = make_deps()
    with (
        patch(_EXTRACTOR) as mock_cls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(
            return_value=[_algo_case(quality_score=0.85)]
        )
        events = await extract_agent_case(event, deps)

    # Exactly one LLM call regardless of agent count.
    assert mock_cls.return_value.aextract.await_count == 1

    # Two writes (one per distinct assistant sender), in first-seen order.
    stored = deps.stores.agent_case.records
    assert [s.scope.owner_id for s in stored] == ["agent_lead", "agent_specialist"]

    # Two events, each tagged with its own agent_id + per-agent entry_id.
    assert [e.agent_id for e in events] == ["agent_lead", "agent_specialist"]
    assert [e.case_entry_id for e in events] == [s.ref.entry_id for s in stored]
    # Same task body / quality across the fan-out (broadcast semantics).
    assert {e.task_intent for e in events} == {"summarise doc"}
    assert {e.quality_score for e in events} == {0.85}

    record = next(e for e in captured if e.get("event") == "agent_case_extracted")
    assert record["owner_ids"] == ["agent_lead", "agent_specialist"]
    assert record["fanout"] == 2


async def test_event_includes_approach_and_key_insight() -> None:
    """1.2.3+ producers populate the case-body fields on AgentCaseExtracted."""
    case = _algo_case(
        task_intent="restore MongoDB replica",
        approach="1. stop node  2. resync  3. verify",
        quality_score=0.75,
        key_insight="watch oplog lag",
    )
    with patch(_EXTRACTOR) as mock_cls:
        mock_cls.return_value.aextract = AsyncMock(return_value=[case])
        (event,) = await extract_agent_case(_event(), make_deps())

    assert event.approach == "1. stop node  2. resync  3. verify"
    assert event.key_insight == "watch oplog lag"
    assert event.task_intent == "restore MongoDB replica"
    assert event.quality_score == 0.75


async def test_empty_key_insight_is_stored_as_absent() -> None:
    deps = make_deps()
    with patch(_EXTRACTOR) as mock_cls:
        mock_cls.return_value.aextract = AsyncMock(
            return_value=[_algo_case(key_insight="")]
        )
        await extract_agent_case(_event(), deps)

    (stored,) = deps.stores.agent_case.records
    assert not stored.item.key_insight
    assert stored.item.task_intent == "summarise doc"


async def test_skips_when_algo_returns_empty() -> None:
    """Algo pre-filter rejected the cell — nothing written, a noop line logged."""
    deps = make_deps()
    with (
        patch(_EXTRACTOR) as mock_cls,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cls.return_value.aextract = AsyncMock(return_value=[])
        events = await extract_agent_case(_event(), deps)

    assert events == []
    assert deps.stores.agent_case.records == []
    assert any(e.get("event") == "agent_case_skipped_by_algo" for e in captured)


async def test_skips_when_no_assistant_sender() -> None:
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
    deps = make_deps()
    with (
        patch(_EXTRACTOR) as mock_cls,
        structlog.testing.capture_logs() as captured,
    ):
        events = await extract_agent_case(event, deps)

    # Algo extractor must not be invoked at all when there's no agent.
    mock_cls.assert_not_called()
    assert events == []
    assert deps.stores.agent_case.records == []
    assert any(e.get("event") == "agent_case_skipped_no_assistant" for e in captured)
