"""Real md round-trip tests: strategy runs → writer writes → reader finds file."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from everalgo.agent_memory import CaseExtractionResult
from everalgo.types import AgentCase, ChatMessage, MemCell

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    MarkdownAgentCaseStore,
)
from everos.memory.deps import MemoryDeps
from everos.memory.events import AgentPipelineStarted
from everos.memory.strategies import extract_agent_case
from tests.fakes import make_deps, make_stores, seed_sources


def _markdown_deps(tmp_path: Path) -> MemoryDeps:
    """Deps whose agent-case store writes real markdown under ``tmp_path``."""
    root = MemoryRoot(root=tmp_path)
    deps = make_deps(stores=make_stores(agent_case=MarkdownAgentCaseStore(root)))
    return seed_sources(deps, memcell_id="mc_a")


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
        mock_ext.return_value.aextract_with_reason = AsyncMock(
            return_value=CaseExtractionResult(cases, None, {})
        )
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
