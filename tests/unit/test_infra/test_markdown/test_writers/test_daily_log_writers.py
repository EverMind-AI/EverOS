"""Tests for the AgentCase daily-log writer, plus the frontmatter timestamp
contract the daily-log writers share.

The daily-log kinds (episode / image memory / agent case) all share
``BaseDailyWriter`` plumbing — exhaustive chassis tests live in
``test_base.py`` and ``test_episode_writer.py`` indirectly via the e2e
flows. Here we focus on the per-kind path resolution + frontmatter shape
that each subclass owns: ``schema``, ``_frontmatter_updates``, and the
writer ↔ reader round-trip on a fresh tmp memory_root.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    AgentCaseReader,
    AgentCaseWriter,
    EpisodeWriter,
    MarkdownReader,
)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


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


async def test_agent_case_writer_appends_multiple(memory_root: MemoryRoot) -> None:
    writer = AgentCaseWriter(memory_root)
    today = _dt.date(2026, 5, 15)
    eid1 = await writer.append_entry(
        "a1",
        inline={
            "owner_id": "a1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "quality_score": 0.5,
        },
        sections={"TaskIntent": "case 1", "Approach": "a"},
        date=today,
    )
    eid2 = await writer.append_entry(
        "a1",
        inline={
            "owner_id": "a1",
            "session_id": "s1",
            "timestamp": "2026-05-15T11:00:00+00:00",
            "parent_id": "mc_2",
            "quality_score": 0.5,
        },
        sections={"TaskIntent": "case 2", "Approach": "a"},
        date=today,
    )
    assert eid1.format() != eid2.format()
    assert eid2.format().endswith("0002")


# ── Display-tz contract for frontmatter timestamps (Gap #5) ────────────


async def test_daily_log_frontmatter_last_appended_at_carries_display_tz_offset(
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

    Checks a user-track writer (``EpisodeWriter``) and the agent-track
    ``AgentCaseWriter`` — they share ``BaseDailyWriter`` plumbing so a
    regression on one would likely affect both, but pinning each rules
    out per-subclass shadowing of ``_frontmatter_updates``.
    """
    from everos.config import load_settings
    from everos.core.utils import datetime as _dt_module

    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    _dt_module._display_tz.cache_clear()

    today = _dt.date(2026, 5, 15)

    # Episode
    ep_writer = EpisodeWriter(memory_root)
    await ep_writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
        },
        sections={"Summary": "x", "Content": "x"},
        date=today,
    )
    ep_path = memory_root.users_dir() / "u1" / "episodes" / "episode-2026-05-15.md"
    ep_fm = (await MarkdownReader.read(ep_path)).frontmatter
    assert ep_fm["last_appended_at"].endswith("+08:00"), ep_fm["last_appended_at"]

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
