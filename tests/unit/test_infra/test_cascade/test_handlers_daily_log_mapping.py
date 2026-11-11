"""``_build_row`` mapping of the agent-case daily-log handler.

The diff loop (read → sha256 → 3-way diff → upsert/delete) lives on
:class:`BaseDailyLogHandler` and is exercised by
``test_handler_episode.py``. These tests focus on the kind-specific
:meth:`_build_row` mapping — given a synthesised ``ParsedEntry``, do
the right LanceDB columns get populated?

The kind gets one happy-path test (all fields present), one with the
optional insight left out, and a focused error-path test (missing
required inline field).
"""

from __future__ import annotations

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import AgentCaseHandler, HandlerDeps
from everos.infra.cascade._projections._daily_log_base import ParsedEntry
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import StructuredEntry
from tests.fakes import handler_deps


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


def _deps(tmp_path) -> HandlerDeps:  # type: ignore[no-untyped-def]
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return handler_deps(
        memory_root=mr,
        tokenizer=_StubTokenizer(),
        embedding=_StubEmbedder(),
    )


def _entry(
    entry_id: str,
    inline: dict[str, str],
    sections: dict[str, str],
    *,
    sha: str = "f" * 64,
) -> ParsedEntry:
    return ParsedEntry(
        entry_id=entry_id,
        structured=StructuredEntry(
            id=entry_id,
            body="",
            start=0,
            end=0,
            header=None,
            inline=inline,
            sections=sections,
        ),
        content_sha256=sha,
    )


async def _embed_one(handler, **kw):  # type: ignore[no-untyped-def]
    """Run one entry through the batch embed path (the production route)."""
    [row] = await handler._embed_entries(
        [kw["entry"]],
        kw["owner_id"],
        kw["owner_type"],
        kw.get("app_id", "default"),
        kw.get("project_id", "default"),
        kw["md_path"],
    )
    return row


# ── AgentCase ────────────────────────────────────────────────────────────


async def test_agent_case_build_row_maps_intent_approach_insight(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = AgentCaseHandler(_deps(tmp_path))
    row = await _embed_one(
        handler,
        owner_id="a1",
        owner_type="agent",
        md_path="agents/a1/.cases/agent_case-2026-05-14.md",
        entry=_entry(
            "ac_20260514_0001",
            inline={
                "owner_id": "a1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "quality_score": "0.87",
            },
            sections={
                "TaskIntent": "scan contract for risk clauses",
                "Approach": "1. read pages 1-5; 2. flag indemnity",
                "KeyInsight": "indemnity cap missing",
            },
        ),
    )
    assert row.task_intent == "scan contract for risk clauses"
    assert row.task_intent_tokens == "scan contract for risk clauses"
    assert row.approach.startswith("1. read pages")
    assert row.key_insight == "indemnity cap missing"
    assert row.quality_score == pytest.approx(0.87)
    assert row.owner_type == "agent"


async def test_agent_case_optional_insight_left_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = AgentCaseHandler(_deps(tmp_path))
    row = await _embed_one(
        handler,
        owner_id="a1",
        owner_type="agent",
        md_path="x.md",
        entry=_entry(
            "ac_20260514_0001",
            inline={
                "owner_id": "a1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "quality_score": "0.5",
            },
            sections={
                "TaskIntent": "x",
                "Approach": "y",
            },
        ),
    )
    assert row.key_insight is None


async def test_agent_case_missing_quality_score_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = AgentCaseHandler(_deps(tmp_path))
    with pytest.raises(ValueError, match="quality_score"):
        await handler._build_row(
            owner_id="a1",
            owner_type="agent",
            md_path="x.md",
            entry=_entry(
                "ac_20260514_0001",
                inline={
                    "owner_id": "a1",
                    "session_id": "s1",
                    "timestamp": "2026-05-14T10:00:00+00:00",
                    "parent_id": "mc_1",
                },
                sections={"TaskIntent": "x", "Approach": "y"},
            ),
        )
