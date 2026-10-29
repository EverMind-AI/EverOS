"""Real md round-trip tests: strategy runs → writer writes → reader finds file."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from everalgo.types import AgentCase, AtomicFact, ChatMessage, Foresight, MemCell

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    AtomicFactReader,
    ForesightReader,
    MarkdownAgentCaseStore,
    MarkdownAtomicFactStore,
    MarkdownForesightStore,
)
from everos.memory.deps import MemoryDeps
from everos.memory.events import (
    AgentPipelineStarted,
    EpisodeExtracted,
    UserPipelineStarted,
)
from everos.memory.strategies import (
    extract_agent_case,
    extract_atomic_facts,
    extract_foresight,
)
from tests.fakes import make_deps, make_stores


def _markdown_deps(tmp_path: Path) -> MemoryDeps:
    """Deps whose daily-log stores write real markdown under ``tmp_path``."""
    root = MemoryRoot(root=tmp_path)
    return make_deps(
        stores=make_stores(
            atomic_fact=MarkdownAtomicFactStore(root),
            foresight=MarkdownForesightStore(root),
            agent_case=MarkdownAgentCaseStore(root),
        )
    )


def _episode_event_for(owner: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_a",
        episode_entry_id="ep_20260517_0001",
        episode_text="hi",
        episode_timestamp_ms=1_700_000_000_000,
        owner_id=owner,
        session_id="s1",
    )


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
) -> None:

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
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_ext,
    ):
        mock_ext.return_value.aextract_from_text = AsyncMock(return_value=facts)
        await extract_atomic_facts(
            _episode_event_for("u_alice"), _markdown_deps(tmp_path)
        )

    reader = AtomicFactReader(root=MemoryRoot(root=tmp_path))
    path = reader.path_for("u_alice")
    assert path.is_file(), f"expected md at {path}"
    content = path.read_text(encoding="utf-8")
    assert "alice likes hiking" in content
    assert "alice lives in tokyo" in content


async def test_foresights_round_trip(
    tmp_path: Path,
) -> None:

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
            "everos.memory.strategies.extract_foresight.ForesightExtractor"
        ) as mock_ext,
    ):
        mock_ext.return_value.aextract = AsyncMock(return_value=foresights)
        await extract_foresight(_event_for("u_alice"), _markdown_deps(tmp_path))

    reader = ForesightReader(root=MemoryRoot(root=tmp_path))
    path = reader.path_for("u_alice")
    assert path.is_file(), f"expected md at {path}"
    content = path.read_text(encoding="utf-8")
    assert "plans trip to tokyo" in content
    assert "said so" in content


async def test_agent_case_round_trip(
    tmp_path: Path,
) -> None:

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
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_ext,
    ):
        mock_ext.return_value.aextract = AsyncMock(return_value=cases)
        await extract_agent_case(_agent_event(), _markdown_deps(tmp_path))

    reader = AgentCaseReader(root=MemoryRoot(root=tmp_path))
    path = reader.path_for("agent_42")
    assert path.is_file(), f"expected md at {path}"
    content = path.read_text(encoding="utf-8")
    assert "summarise the doc" in content
    assert "read then condense" in content
    assert "batch-then-summarise" in content
    # quality_score must land in inline (cascade requires it via require_float).
    assert "quality_score" in content
