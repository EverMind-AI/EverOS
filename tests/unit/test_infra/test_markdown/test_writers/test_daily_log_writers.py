"""Tests for AtomicFact / Foresight / AgentCase daily-log writers.

The 4 daily-log kinds (episode + these 3) all share ``BaseDailyWriter``
plumbing — exhaustive chassis tests live in ``test_base.py`` and
``test_episode_writer.py`` indirectly via the e2e flows. Here we focus
on the per-kind path resolution + frontmatter shape that each
subclass owns: ``schema``, ``_frontmatter_updates``, and the
writer ↔ reader round-trip on a fresh tmp memory_root.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    AgentCaseWriter,
    AtomicFactReader,
    AtomicFactWriter,
    ForesightReader,
    ForesightWriter,
)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


# ── AtomicFact ────────────────────────────────────────────────────────────


async def test_atomic_fact_writer_round_trip(memory_root: MemoryRoot) -> None:
    writer = AtomicFactWriter(memory_root)
    today = _dt.date(2026, 5, 15)
    eid = await writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
        },
        sections={"Fact": "Alice prefers Italian."},
        date=today,
    )
    path = (
        memory_root.users_dir() / "u1" / ".atomic_facts" / "atomic_fact-2026-05-15.md"
    )
    parsed = await MarkdownReader.read(path)

    # frontmatter
    fm = parsed.frontmatter
    assert fm["id"] == "atomic_fact_log_u1_2026-05-15"
    assert fm["type"] == "atomic_fact_daily"
    assert fm["file_type"] == "atomic_fact_daily"
    assert fm["user_id"] == "u1"
    assert fm["track"] == "user"
    assert fm["date"] == "2026-05-15"
    assert fm["entry_count"] == 1

    # entry body
    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.id == eid.format()
    structured = entry.as_structured()
    assert structured.inline["owner_id"] == "u1"
    assert structured.inline["parent_id"] == "mc_1"
    assert structured.sections["Fact"] == "Alice prefers Italian."

    # reader is symmetric
    reader = AtomicFactReader(memory_root)
    assert reader.path_for("u1", today) == path
    found = await reader.find_structured("u1", eid)
    assert found is not None
    assert found.sections["Fact"] == "Alice prefers Italian."


async def test_atomic_fact_writer_appends_multiple(memory_root: MemoryRoot) -> None:
    writer = AtomicFactWriter(memory_root)
    today = _dt.date(2026, 5, 15)
    eid1 = await writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
        },
        sections={"Fact": "fact 1"},
        date=today,
    )
    eid2 = await writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T11:00:00+00:00",
            "parent_id": "mc_2",
        },
        sections={"Fact": "fact 2"},
        date=today,
    )
    assert eid1.format() != eid2.format()
    assert eid2.format().endswith("0002")


# ── Foresight ─────────────────────────────────────────────────────────────


async def test_foresight_writer_round_trip(memory_root: MemoryRoot) -> None:
    writer = ForesightWriter(memory_root)
    today = _dt.date(2026, 5, 15)
    eid = await writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "start_time": "2026-05-15T12:00:00+00:00",
            "end_time": "2026-05-15T13:00:00+00:00",
            "duration_days": 1,
        },
        sections={
            "Foresight": "User will book lunch at noon.",
            "Evidence": "Past calendar pattern.",
        },
        date=today,
    )
    path = memory_root.users_dir() / "u1" / ".foresights" / "foresight-2026-05-15.md"
    parsed = await MarkdownReader.read(path)
    fm = parsed.frontmatter
    assert fm["id"] == "foresight_log_u1_2026-05-15"
    assert fm["type"] == "foresight_daily"

    structured = parsed.entries[0].as_structured()
    assert structured.sections["Foresight"] == "User will book lunch at noon."
    assert structured.sections["Evidence"] == "Past calendar pattern."
    assert structured.inline["duration_days"] == "1"
    assert structured.inline["start_time"].startswith("2026-05-15T12:00:00")

    reader = ForesightReader(memory_root)
    found = await reader.find_structured("u1", eid)
    assert found is not None
    assert found.sections["Evidence"] == "Past calendar pattern."


# ── AgentCase ─────────────────────────────────────────────────────────────


async def test_agent_case_writer_round_trip(memory_root: MemoryRoot) -> None:
    writer = AgentCaseWriter(memory_root)
    today = _dt.date(2026, 5, 15)
    eid = await writer.append_entry(
        "a1",
        inline={
            "owner_id": "a1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_agent",
            "quality_score": 0.87,
        },
        sections={
            "TaskIntent": "Scan contract for indemnity gaps.",
            "Approach": "1. read sections;\n2. flag clauses;\n3. cross-check cap.",
            "KeyInsight": "Indemnity cap missing in section 4.",
        },
        date=today,
    )
    path = memory_root.agents_dir() / "a1" / ".cases" / "agent_case-2026-05-15.md"
    parsed = await MarkdownReader.read(path)
    fm = parsed.frontmatter
    assert fm["id"] == "agent_case_log_a1_2026-05-15"
    assert fm["type"] == "agent_case_daily"
    assert fm["agent_id"] == "a1"
    assert fm["track"] == "agent"

    structured = parsed.entries[0].as_structured()
    assert structured.inline["quality_score"] == "0.87"
    assert structured.sections["TaskIntent"].startswith("Scan contract")
    assert structured.sections["Approach"].startswith("1. read sections")
    assert structured.sections["KeyInsight"].startswith("Indemnity cap missing")

    reader = AgentCaseReader(memory_root)
    assert reader.path_for("a1", today) == path
    found = await reader.find_structured("a1", eid)
    assert found is not None
    assert found.sections["TaskIntent"].startswith("Scan contract")


# ── round-trip with cascade handler (md → LanceDB row mapping) ─────────────


async def test_atomic_fact_writer_output_feeds_handler(
    memory_root: MemoryRoot,
) -> None:
    """The writer's md is exactly what AtomicFactHandler expects to read."""
    from everos.component.embedding import EmbeddingProvider
    from everos.component.tokenizer import Tokenizer
    from everos.memory.cascade.handlers import AtomicFactHandler, HandlerDeps
    from everos.memory.cascade.handlers._daily_log_base import ParsedEntry

    class _T(Tokenizer):
        def tokenize(self, t):  # type: ignore[no-untyped-def]
            return [x for x in t.split() if x]

        def tokenize_batch(self, ts):  # type: ignore[no-untyped-def]
            return [self.tokenize(x) for x in ts]

    class _E(EmbeddingProvider):
        dim = 1024

        async def embed(self, t):  # type: ignore[no-untyped-def]
            return [0.0] * self.dim

        async def embed_batch(self, ts):  # type: ignore[no-untyped-def]
            return [await self.embed(x) for x in ts]

    today = _dt.date(2026, 5, 15)
    eid = await AtomicFactWriter(memory_root).append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
        },
        sections={"Fact": "Alice prefers Italian."},
        date=today,
    )
    path = (
        memory_root.users_dir() / "u1" / ".atomic_facts" / "atomic_fact-2026-05-15.md"
    )
    rel = path.relative_to(memory_root.root).as_posix()
    parsed = await MarkdownReader.read(path)
    entry = parsed.entries[0]
    handler = AtomicFactHandler(
        HandlerDeps(memory_root=memory_root, embedder=_E(), tokenizer=_T())
    )
    structured = entry.as_structured()
    pe = ParsedEntry(entry.id, structured, handler._content_sha256(structured))
    row = await handler._build_row(
        owner_id="u1", owner_type="user", md_path=rel, entry=pe
    )
    assert row.id == f"u1_{eid.format()}"
    assert row.fact == "Alice prefers Italian."
    assert row.parent_id == "mc_1"
    assert row.sender_ids == ["u1"]
    assert len(row.vector) == 1024


# ── Display-tz contract for frontmatter timestamps (Gap #5) ────────────


async def test_atomic_fact_frontmatter_last_appended_at_carries_display_tz_offset(
    memory_root: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``last_appended_at`` in markdown frontmatter renders in the display tz.

    Markdown frontmatter is a display-side artefact (users read the file
    directly), so ``last_appended_at`` must use
    :func:`get_now_with_timezone` not :func:`get_utc_now`. Pins that
    contract end-to-end: configure ``EVEROS_MEMORY__TIMEZONE=Asia/Shanghai``,
    write an entry, read the .md file, assert the literal string ends
    with ``+08:00``.

    Repeats the same check for ``ForesightWriter`` and
    ``AgentCaseWriter`` — they share ``BaseDailyWriter`` plumbing so a
    regression on one would likely affect all three, but pinning each
    rules out per-subclass shadowing of ``_frontmatter_updates``.
    """
    from everos.component.utils import datetime as _dt_module
    from everos.config import load_settings

    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    _dt_module._display_tz.cache_clear()

    today = _dt.date(2026, 5, 15)

    # AtomicFact
    af_writer = AtomicFactWriter(memory_root)
    await af_writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
        },
        sections={"Fact": "x"},
        date=today,
    )
    af_path = (
        memory_root.users_dir() / "u1" / ".atomic_facts" / "atomic_fact-2026-05-15.md"
    )
    af_fm = (await MarkdownReader.read(af_path)).frontmatter
    assert af_fm["last_appended_at"].endswith("+08:00"), af_fm["last_appended_at"]

    # Foresight
    fs_writer = ForesightWriter(memory_root)
    await fs_writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "scope": "today",
            "horizon_days": 1,
        },
        sections={"Foresight": "x"},
        date=today,
    )
    fs_path = memory_root.users_dir() / "u1" / ".foresights" / "foresight-2026-05-15.md"
    fs_fm = (await MarkdownReader.read(fs_path)).frontmatter
    assert fs_fm["last_appended_at"].endswith("+08:00"), fs_fm["last_appended_at"]

    # AgentCase
    ac_writer = AgentCaseWriter(memory_root)
    await ac_writer.append_entry(
        "a1",
        inline={
            "owner_id": "a1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "quality_score": 0.9,
        },
        sections={"Task intent": "x", "Approach": "y"},
        date=today,
    )
    ac_path = memory_root.agents_dir() / "a1" / ".cases" / "agent_case-2026-05-15.md"
    ac_fm = (await MarkdownReader.read(ac_path)).frontmatter
    assert ac_fm["last_appended_at"].endswith("+08:00"), ac_fm["last_appended_at"]
