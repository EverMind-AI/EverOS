"""Per-kind ``_build_row`` mapping for the 3 non-Episode daily-log handlers.

The diff loop (read → sha256 → 3-way diff → upsert/delete) lives on
:class:`BaseDailyLogHandler` and is exercised by
``test_handler_episode.py``. These tests focus on the kind-specific
:meth:`_build_row` mapping — given a synthesised ``ParsedEntry``, do
the right LanceDB columns get populated?

Each kind gets one happy-path test (all fields present) plus a
focused error-path test (missing required inline field). Sharing one
file avoids 3 nearly-identical fixture stacks.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot, StructuredEntry
from everos.memory.cascade.handlers import (
    AgentCaseHandler,
    AtomicFactHandler,
    ForesightHandler,
    HandlerDeps,
)
from everos.memory.cascade.handlers._daily_log_base import ParsedEntry


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
    return HandlerDeps(
        memory_root=mr,
        embedder=_StubEmbedder(),
        tokenizer=_StubTokenizer(),
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


# ── AtomicFact ───────────────────────────────────────────────────────────


async def test_atomic_fact_build_row_maps_inline_and_section(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = AtomicFactHandler(_deps(tmp_path))
    row = await handler._build_row(
        owner_id="u1",
        owner_type="user",
        md_path="users/u1/.atomic_facts/atomic_fact-2026-05-14.md",
        entry=_entry(
            "af_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "sender_ids": "[u1, u2]",
            },
            sections={"Fact": "the user prefers dark mode"},
        ),
    )
    assert row.id == "u1_af_20260514_0001"
    assert row.fact == "the user prefers dark mode"
    assert row.fact_tokens == "the user prefers dark mode"
    assert row.parent_id == "mc_1"
    assert row.sender_ids == ["u1", "u2"]
    assert row.timestamp == _dt.datetime(2026, 5, 14, 10, 0, tzinfo=_dt.UTC)
    assert row.md_path.endswith("atomic_fact-2026-05-14.md")
    assert len(row.vector) == 1024


async def test_atomic_fact_missing_timestamp_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = AtomicFactHandler(_deps(tmp_path))
    with pytest.raises(ValueError, match="timestamp"):
        await handler._build_row(
            owner_id="u1",
            owner_type="user",
            md_path="x.md",
            entry=_entry(
                "af_20260514_0001",
                inline={"owner_id": "u1", "session_id": "s1"},
                sections={"Fact": "x"},
            ),
        )


# ── Foresight ────────────────────────────────────────────────────────────


async def test_foresight_build_row_with_evidence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = ForesightHandler(_deps(tmp_path))
    row = await handler._build_row(
        owner_id="u1",
        owner_type="user",
        md_path="users/u1/.foresights/foresight-2026-05-14.md",
        entry=_entry(
            "fs_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "start_time": "2026-05-14T11:00:00+00:00",
                "end_time": "2026-05-14T13:00:00+00:00",
                "duration_days": "2",
            },
            sections={
                "Foresight": "user will book lunch",
                "Evidence": "calendar invite mentions 12pm",
            },
        ),
    )
    assert row.foresight == "user will book lunch"
    assert row.foresight_tokens == "user will book lunch"
    assert row.evidence == "calendar invite mentions 12pm"
    assert row.evidence_tokens == "calendar invite mentions 12pm"
    assert row.start_time == _dt.datetime(2026, 5, 14, 11, 0, tzinfo=_dt.UTC)
    assert row.end_time == _dt.datetime(2026, 5, 14, 13, 0, tzinfo=_dt.UTC)
    assert row.duration_days == 2


async def test_foresight_optional_evidence_left_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = ForesightHandler(_deps(tmp_path))
    row = await handler._build_row(
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=_entry(
            "fs_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
            },
            sections={"Foresight": "user will book lunch"},
        ),
    )
    assert row.evidence is None
    assert row.evidence_tokens is None
    assert row.start_time is None
    assert row.end_time is None
    assert row.duration_days is None


# ── AgentCase ────────────────────────────────────────────────────────────


async def test_agent_case_build_row_maps_intent_approach_insight(tmp_path) -> None:  # type: ignore[no-untyped-def]
    handler = AgentCaseHandler(_deps(tmp_path))
    row = await handler._build_row(
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
    row = await handler._build_row(
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
