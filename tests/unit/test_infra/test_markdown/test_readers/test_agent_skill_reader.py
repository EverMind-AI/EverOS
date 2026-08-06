"""Tests for :class:`AgentSkillReader` — typed read for the skill directory layout."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillReader,
    AgentSkillWriter,
)


def _make_fm(**overrides: object) -> AgentSkillFrontmatter:
    base: dict[str, object] = {
        "id": "agent_x_skill_alpha",
        "agent_id": "agent_x",
        "name": "alpha",
        "description": "A test skill.",
        "confidence": 0.5,
        "maturity_score": 0.5,
    }
    base.update(overrides)
    return AgentSkillFrontmatter(**base)  # type: ignore[arg-type]


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


@pytest.fixture
def writer(root: MemoryRoot) -> AgentSkillWriter:
    return AgentSkillWriter(root)


@pytest.fixture
def reader(root: MemoryRoot) -> AgentSkillReader:
    return AgentSkillReader(root)


async def test_read_main_returns_typed_frontmatter_and_body(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    fm_in = _make_fm(
        description="Contract risk scan.",
        confidence=0.88,
        maturity_score=0.82,
        source_case_ids=["case_a", "case_b"],
    )
    await writer.write_main("agent_x", "alpha", frontmatter=fm_in, body="The body.")

    out = await reader.read_main("agent_x", "alpha", schema=AgentSkillFrontmatter)
    assert out is not None
    fm_out, body = out
    assert isinstance(fm_out, AgentSkillFrontmatter)
    assert fm_out.name == "alpha"
    assert fm_out.source_case_ids == ["case_a", "case_b"]
    assert fm_out.confidence == 0.88
    assert fm_out.maturity_score == 0.82
    assert body == "The body."


async def test_read_main_returns_none_when_missing(reader: AgentSkillReader) -> None:
    assert (
        await reader.read_main("agent_x", "ghost", schema=AgentSkillFrontmatter) is None
    )


async def test_read_main_round_trip_through_extra_fields(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    """L2 / L4 ride-along fields survive a write+read cycle (extra="allow")."""
    fm_in = _make_fm(md_sha256="abc", custom_label="ride-along")
    await writer.write_main("agent_x", "alpha", frontmatter=fm_in, body="b")
    out = await reader.read_main("agent_x", "alpha", schema=AgentSkillFrontmatter)
    assert out is not None
    fm_out, _ = out
    dumped = fm_out.model_dump()
    assert dumped["md_sha256"] == "abc"
    assert dumped["custom_label"] == "ride-along"


async def test_read_main_validates_against_supplied_schema(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    """A stricter schema rejects loose existing data — proves typed parsing."""

    class _StricterSkillFM(AgentSkillFrontmatter):
        # Required field with no default — written file lacks it.
        priority: int

    fm_in = _make_fm()
    await writer.write_main("agent_x", "alpha", frontmatter=fm_in, body="b")

    with pytest.raises(ValidationError):
        await reader.read_main("agent_x", "alpha", schema=_StricterSkillFM)


async def test_read_reference_round_trip(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    await writer.write_reference(
        "agent_x", "alpha", "termination", "## term clauses\n..."
    )
    content = await reader.read_reference("agent_x", "alpha", "termination")
    assert content == "## term clauses\n..."


async def test_read_reference_returns_none_when_missing(
    reader: AgentSkillReader,
) -> None:
    assert await reader.read_reference("agent_x", "alpha", "ghost") is None


async def test_read_script_round_trip(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    await writer.write_script("agent_x", "alpha", "redline.py", "print('hi')\n")
    content = await reader.read_script("agent_x", "alpha", "redline.py")
    assert content == "print('hi')"


async def test_read_script_returns_none_when_missing(reader: AgentSkillReader) -> None:
    assert await reader.read_script("agent_x", "alpha", "ghost.py") is None


async def test_list_by_cluster_returns_matching_skills(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    """Enumerates SKILL.md under the agent, filters by frontmatter cluster_id."""
    await writer.write_main(
        "a1",
        "revive_replica",
        frontmatter=_make_fm(
            id="a1_revive_replica",
            agent_id="a1",
            name="revive_replica",
            cluster_id="cl1",
        ),
        body="b",
    )
    await writer.write_main(
        "a1",
        "drain_queue",
        frontmatter=_make_fm(
            id="a1_drain_queue", agent_id="a1", name="drain_queue", cluster_id="cl1"
        ),
        body="b",
    )
    await writer.write_main(
        "a1",
        "rotate_secrets",
        frontmatter=_make_fm(
            id="a1_rotate_secrets",
            agent_id="a1",
            name="rotate_secrets",
            cluster_id="cl2",
        ),
        body="b",
    )

    results = await reader.list_by_cluster("a1", "cl1")

    names = sorted(r.name for r in results)
    assert names == ["drain_queue", "revive_replica"]


async def test_list_by_cluster_ignores_skills_without_cluster_id(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    """Skills whose frontmatter cluster_id is None never leak into any bucket."""
    await writer.write_main(
        "a1",
        "orphan",
        frontmatter=_make_fm(id="a1_orphan", agent_id="a1", name="orphan"),
        body="b",
    )
    await writer.write_main(
        "a1",
        "assigned",
        frontmatter=_make_fm(
            id="a1_assigned", agent_id="a1", name="assigned", cluster_id="cl1"
        ),
        body="b",
    )

    assert [r.name for r in await reader.list_by_cluster("a1", "cl1")] == ["assigned"]
    assert await reader.list_by_cluster("a1", "cl_missing") == []


async def test_list_by_cluster_missing_dir_returns_empty(
    reader: AgentSkillReader,
) -> None:
    """New agent with no skill dir yet — returns [] without raising."""
    assert await reader.list_by_cluster("a_new", "cl1") == []


async def test_frontmatter_name_and_directory_name_agree_after_sanitization(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    """``list_by_cluster`` (directory-derived name) and
    ``_hydrate_algo_skills``-style re-read (raw ``fm.name``) must land on
    the same file.

    This is the exact seam sanitization would break if the writer and
    reader derived the ``skill_<name>`` segment independently:
    ``list_by_cluster`` recovers ``skill_name`` from the on-disk
    (already-sanitized) directory, while
    ``memory.strategies.extract_agent_skill._hydrate_algo_skills`` re-reads
    using the frontmatter's raw ``name`` field. Routing both through the
    same ``skill_dir_name`` classmethod keeps them consistent because
    sanitization is idempotent.
    """
    space_name = "修复 Django 自动重载问题"
    await writer.write_main(
        "a1",
        space_name,
        frontmatter=_make_fm(
            id="a1_django_reload_fix",
            agent_id="a1",
            name=space_name,
            cluster_id="cl1",
        ),
        body="The fix body.",
    )

    listed = await reader.list_by_cluster("a1", "cl1")
    assert len(listed) == 1
    fm = listed[0]
    assert fm.name == space_name

    # Re-derive the path from the raw frontmatter name, exactly like
    # ``_hydrate_algo_skills`` does — must resolve to the same file.
    out = await reader.read_main("a1", fm.name, schema=AgentSkillFrontmatter)
    assert out is not None
    fm_out, body = out
    assert fm_out.name == space_name
    assert body == "The fix body."
