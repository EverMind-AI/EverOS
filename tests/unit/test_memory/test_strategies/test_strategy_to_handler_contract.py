"""Contract: strategy-written md must round-trip through cascade handler.

Guards against silent-breakage class: strategy writes section keys
(e.g. ``{"task_intent": ...}``) that the cascade handler reads under a
different case (e.g. ``sections.get("TaskIntent")``). Without this contract,
the worker still upserts a LanceDB row but with empty ``task_intent`` /
``approach`` text, empty BM25 tokens, and a vector for the empty string —
search fails silently. Earlier unit tests stop at the strategy boundary (mock
the writer) or at the writer boundary (skip the strategy); neither
catches a key-name drift.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
from everalgo.agent_memory import CaseExtractionResult
from everalgo.types import AgentCase, ChatMessage, MemCell

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import AgentCaseHandler
from everos.infra.cascade._projections._daily_log_base import ParsedEntry
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownAgentCaseStore,
    MarkdownReader,
)
from everos.memory.deps import MemoryDeps
from everos.memory.events import AgentPipelineStarted
from everos.memory.strategies import extract_agent_case
from tests.fakes import handler_deps, make_deps, make_stores, seed_sources


class _StubTokenizer(Tokenizer):
    def tokenize(self, text):  # type: ignore[no-untyped-def]
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text):  # type: ignore[no-untyped-def]
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


def _markdown_deps(tmp_path: Path) -> MemoryDeps:
    """Deps whose agent-case store writes real markdown under ``tmp_path``."""
    root = MemoryRoot(root=tmp_path)
    deps = make_deps(stores=make_stores(agent_case=MarkdownAgentCaseStore(root)))
    return seed_sources(deps, memcell_id="mc_a")


async def _build_row_from_md(
    handler: AgentCaseHandler,
    md_root: Path,
    md_glob: str,
    *,
    owner_id: str = "u_alice",
    owner_type: str = "user",
):
    md_files: list[anyio.Path] = []
    async for p in anyio.Path(md_root).glob(md_glob):
        md_files.append(p)
    assert len(md_files) == 1, f"expected exactly one md, got: {md_files}"
    md_abs = Path(md_files[0])
    rel = str(md_abs.relative_to(md_root))
    parsed = await MarkdownReader.read(md_abs)
    assert parsed.entries, "writer should have produced at least one entry"
    entry = parsed.entries[0]
    structured = entry.as_structured()
    pe = ParsedEntry(
        entry_id=entry.id,
        structured=structured,
        content_sha256=handler._content_sha256(structured),
    )
    [row] = await handler._embed_entries(
        [pe], owner_id, owner_type, "default", "default", rel
    )
    return row


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


async def test_agent_case_strategy_md_feeds_handler_with_content(
    tmp_path: Path,
) -> None:
    """Strategy → md → AgentCaseHandler carries task_intent, approach, score."""

    cases = [
        AgentCase(
            id=uuid.uuid4().hex,
            timestamp=1_700_000_001_000,
            task_intent="summarise the doc",
            approach="read + condense",
            quality_score=0.85,
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

    handler = AgentCaseHandler(
        handler_deps(
            memory_root=MemoryRoot(root=tmp_path),
            tokenizer=_StubTokenizer(),
            embedding=_StubEmbedder(),
        )
    )
    row = await _build_row_from_md(
        handler,
        tmp_path,
        "*/*/agents/agent_42/.cases/agent_case-*.md",
        owner_id="agent_42",
        owner_type="agent",
    )
    # Regression guard: section-key drift or missing quality_score inline
    # would surface as empty strings / require_float failure.
    assert row.task_intent == "summarise the doc"
    assert row.task_intent_tokens == "summarise the doc"
    assert row.approach == "read + condense"
    assert row.approach_tokens == "read + condense"
    assert row.key_insight == "batch-then-summarise"
    assert row.quality_score == 0.85
    assert row.owner_id == "agent_42"
    assert row.owner_type == "agent"
    assert len(row.vector) == 1024
