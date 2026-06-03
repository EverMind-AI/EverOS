"""Real md round-trip tests: strategy runs → writer writes → reader finds file."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import AgentCase, AtomicFact, ChatMessage, Foresight, MemCell

from everos.core.persistence import MemoryRoot
from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    AtomicFactReader,
    ForesightReader,
)
from everos.memory.events import AgentPipelineStarted, UserPipelineStarted
from everos.memory.strategies.extract_agent_case import extract_agent_case
from everos.memory.strategies.extract_atomic_facts import extract_atomic_facts
from everos.memory.strategies.extract_foresight import extract_foresight


def _event_for(owner: str) -> UserPipelineStarted:
    return UserPipelineStarted(
        memcell_id="mc_a",
        session_id="s1",
        memcell=MemCell(
            items=[
                ChatMessage(
                    id="m1",
                    role="user",
                    content="hi",
                    timestamp=1_700_000_000_000,
                    sender_id=owner,
                ),
            ],
            timestamp=1_700_000_000_000,
        ),
    )


def _agent_event() -> AgentPipelineStarted:
    return AgentPipelineStarted(
        memcell_id="mc_a",
        session_id="s1",
        memcell=MemCell(
            items=[
                ChatMessage(
                    id="m1",
                    role="user",
                    content="please summarise",
                    timestamp=1_700_000_000_000,
                    sender_id="u_alice",
                ),
                ChatMessage(
                    id="m2",
                    role="assistant",
                    content="here's the summary",
                    timestamp=1_700_000_001_000,
                    sender_id="agent_42",
                ),
            ],
            timestamp=1_700_000_001_000,
        ),
    )


async def test_atomic_facts_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(af_mod, "_writer", None, raising=False)

    facts = [
        AtomicFact(
            owner_id="u_alice",
            content="alice likes hiking",
            timestamp=1_700_000_000_000,
        ),
        AtomicFact(
            owner_id="u_alice",
            content="alice lives in tokyo",
            timestamp=1_700_000_000_000,
        ),
    ]

    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_ext,
    ):
        mock_ext.return_value.aextract = AsyncMock(return_value=facts)
        await extract_atomic_facts(_event_for("u_alice"), FakeStrategyContext())

    reader = AtomicFactReader(root=MemoryRoot(root=tmp_path))
    path = reader.path_for("u_alice")
    assert path.is_file(), f"expected md at {path}"
    content = path.read_text(encoding="utf-8")
    assert "alice likes hiking" in content
    assert "alice lives in tokyo" in content


async def test_foresights_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(fs_mod, "_writer", None, raising=False)

    foresights = [
        Foresight(
            owner_id="u_alice",
            foresight="plans trip to tokyo",
            evidence="said so",
            timestamp=1_700_000_000_000,
        ),
    ]

    with (
        patch(
            "everos.memory.strategies.extract_foresight.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_foresight.ForesightExtractor"
        ) as mock_ext,
    ):
        mock_ext.return_value.aextract = AsyncMock(return_value=foresights)
        await extract_foresight(_event_for("u_alice"), FakeStrategyContext())

    reader = ForesightReader(root=MemoryRoot(root=tmp_path))
    path = reader.path_for("u_alice")
    assert path.is_file(), f"expected md at {path}"
    content = path.read_text(encoding="utf-8")
    assert "plans trip to tokyo" in content
    assert "said so" in content


async def test_agent_case_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    ac_mod = importlib.import_module("everos.memory.strategies.extract_agent_case")

    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(ac_mod, "_writer", None, raising=False)

    cases = [
        AgentCase(
            id=uuid.uuid4().hex,
            timestamp=1_700_000_001_000,
            task_intent="summarise the doc",
            approach="read then condense",
            quality_score=0.82,
            key_insight="batch-then-summarise",
        )
    ]

    with (
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_ext,
    ):
        mock_ext.return_value.aextract = AsyncMock(return_value=cases)
        await extract_agent_case(_agent_event(), FakeStrategyContext())

    reader = AgentCaseReader(root=MemoryRoot(root=tmp_path))
    path = reader.path_for("agent_42")
    assert path.is_file(), f"expected md at {path}"
    content = path.read_text(encoding="utf-8")
    assert "summarise the doc" in content
    assert "read then condense" in content
    assert "batch-then-summarise" in content
    # quality_score must land in inline (cascade requires it via require_float).
    assert "quality_score" in content
