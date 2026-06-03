"""Tests for :class:`AgentSkillWriter` — directory + progressive disclosure."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillWriter,
)


def _make_fm(**overrides: object) -> AgentSkillFrontmatter:
    """Build an AgentSkillFrontmatter with sensible defaults for tests."""
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


async def test_write_main_creates_directory_layout(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    fm = _make_fm()
    path = await writer.write_main(
        "agent_x", "alpha", frontmatter=fm, body="Step 1: do thing."
    )
    expected = root.agents_dir() / "agent_x" / "skills" / "skill_alpha" / "SKILL.md"
    assert path == expected
    assert expected.is_file()


async def test_write_main_writes_frontmatter_and_body(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    fm = _make_fm(
        description="Contract risk scan.",
        confidence=0.88,
        maturity_score=0.82,
        source_case_ids=["case_a", "case_b"],
        cluster_id="cl_x",
    )
    await writer.write_main("agent_x", "alpha", frontmatter=fm, body="The body.")
    parsed = await MarkdownReader.read(
        root.agents_dir() / "agent_x" / "skills" / "skill_alpha" / "SKILL.md"
    )
    assert parsed.frontmatter["name"] == "alpha"
    assert parsed.frontmatter["description"] == "Contract risk scan."
    assert parsed.frontmatter["confidence"] == 0.88
    assert parsed.frontmatter["maturity_score"] == 0.82
    assert parsed.frontmatter["source_case_ids"] == ["case_a", "case_b"]
    assert parsed.frontmatter["cluster_id"] == "cl_x"
    assert parsed.body.rstrip("\n") == "The body."


async def test_write_main_is_upsert_full_replace(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    """Second call overwrites both frontmatter and body — no append."""
    fm1 = _make_fm(description="v1", maturity_score=0.4)
    await writer.write_main("agent_x", "alpha", frontmatter=fm1, body="body v1")

    fm2 = _make_fm(description="v2", maturity_score=0.7)
    await writer.write_main("agent_x", "alpha", frontmatter=fm2, body="body v2")

    parsed = await MarkdownReader.read(
        root.agents_dir() / "agent_x" / "skills" / "skill_alpha" / "SKILL.md"
    )
    assert parsed.frontmatter["description"] == "v2"
    assert parsed.frontmatter["maturity_score"] == 0.7
    assert parsed.body.rstrip("\n") == "body v2"
    # No "body v1" residue from the previous version.
    assert "body v1" not in parsed.body


async def test_write_reference_uses_md_extension(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    path = await writer.write_reference(
        "agent_x", "alpha", "termination_clauses", "## Termination\n..."
    )
    expected = (
        root.agents_dir()
        / "agent_x"
        / "skills"
        / "skill_alpha"
        / "references"
        / "termination_clauses.md"
    )
    assert path == expected
    assert path.read_text(encoding="utf-8").startswith("## Termination")


async def test_write_script_keeps_full_filename(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    path = await writer.write_script("agent_x", "alpha", "redline.py", "print('hi')\n")
    expected = (
        root.agents_dir()
        / "agent_x"
        / "skills"
        / "skill_alpha"
        / "scripts"
        / "redline.py"
    )
    assert path == expected
    assert path.read_text(encoding="utf-8") == "print('hi')\n"


def test_main_path_does_not_create_anything(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    """``main_path`` is a pure path resolver — no IO."""
    p = writer.main_path("agent_x", "alpha")
    assert p.name == "SKILL.md"
    assert not root.agents_dir().exists()


async def test_write_main_normalises_trailing_newline(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    """Body without a trailing newline still ends in exactly one newline."""
    fm = _make_fm()
    await writer.write_main("agent_x", "alpha", frontmatter=fm, body="no-newline-end")
    text = (
        root.agents_dir() / "agent_x" / "skills" / "skill_alpha" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert text.endswith("no-newline-end\n")
