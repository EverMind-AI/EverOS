"""Contract: strategy-written md must round-trip through cascade handler.

Guards against silent-breakage class: strategy writes section keys
(e.g. ``{"fact": ...}``) that the cascade handler reads under a different
case (e.g. ``sections.get("Fact")``). Without this contract, the worker
still upserts a LanceDB row but with empty ``fact`` / ``foresight``
text, empty BM25 tokens, and a vector for the empty string — search
fails silently. Earlier unit tests stop at the strategy boundary (mock
the writer) or at the writer boundary (skip the strategy); neither
catches a key-name drift.
"""

from __future__ import annotations

import importlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from everalgo.types import AgentCase, AtomicFact, ChatMessage, Foresight, MemCell

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.ome.testing import FakeStrategyContext
from everos.memory.cascade.handlers import (
    AgentCaseHandler,
    AtomicFactHandler,
    ForesightHandler,
    HandlerDeps,
)
from everos.memory.cascade.handlers._daily_log_base import ParsedEntry
from everos.memory.events import (
    AgentPipelineStarted,
    EpisodeExtracted,
    UserPipelineStarted,
)
from everos.memory.strategies.extract_agent_case import extract_agent_case
from everos.memory.strategies.extract_atomic_facts import extract_atomic_facts
from everos.memory.strategies.extract_foresight import extract_foresight


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


def _episode_event(owner_id: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_a",
        episode_entry_id="ep_20260517_0001",
        episode_text="hi",
        episode_timestamp_ms=1_700_000_000_000,
        owner_id=owner_id,
        session_id="s1",
    )


def _event(owner_id: str) -> UserPipelineStarted:
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
                    sender_id=owner_id,
                ),
            ],
            timestamp=1_700_000_000_000,
        ),
    )


async def _build_row_from_md(
    handler: AtomicFactHandler | ForesightHandler | AgentCaseHandler,
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
        content_sha256=handler._content_sha256(structured),  # noqa: SLF001
    )
    return await handler._build_row(  # noqa: SLF001
        owner_id=owner_id,
        owner_type=owner_type,
        md_path=rel,
        entry=pe,
    )


async def test_atomic_fact_strategy_md_feeds_handler_with_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strategy → md → AtomicFactHandler must carry the fact text intact."""
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
        mock_ext.return_value.aextract_from_text = AsyncMock(return_value=facts)
        await extract_atomic_facts(_episode_event("u_alice"), FakeStrategyContext())

    handler = AtomicFactHandler(
        HandlerDeps(
            memory_root=MemoryRoot(root=tmp_path),
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    row = await _build_row_from_md(
        handler, tmp_path, "*/*/users/u_alice/.atomic_facts/atomic_fact-*.md"
    )
    # Regression guard: section key drift would land here as fact="".
    assert row.fact == "alice likes hiking"
    assert row.fact_tokens == "alice likes hiking"
    assert len(row.vector) == 1024


async def test_foresight_strategy_md_feeds_handler_with_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strategy → md → ForesightHandler must carry foresight + evidence text."""
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(fs_mod, "_writer", None, raising=False)

    foresights = [
        Foresight(
            owner_id="u_alice",
            foresight="plans trip to tokyo",
            evidence="said so explicitly",
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
        await extract_foresight(_event("u_alice"), FakeStrategyContext())

    handler = ForesightHandler(
        HandlerDeps(
            memory_root=MemoryRoot(root=tmp_path),
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    row = await _build_row_from_md(
        handler, tmp_path, "*/*/users/u_alice/.foresights/foresight-*.md"
    )
    # Regression guard: section key drift would land here as foresight="".
    assert row.foresight == "plans trip to tokyo"
    assert row.foresight_tokens == "plans trip to tokyo"
    assert row.evidence == "said so explicitly"
    assert row.evidence_tokens == "said so explicitly"
    assert len(row.vector) == 1024


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strategy → md → AgentCaseHandler carries task_intent, approach, score."""
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
            approach="read + condense",
            quality_score=0.85,
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

    handler = AgentCaseHandler(
        HandlerDeps(
            memory_root=MemoryRoot(root=tmp_path),
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
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
