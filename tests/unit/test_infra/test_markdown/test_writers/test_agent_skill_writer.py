"""Tests for :class:`AgentSkillWriter` — directory + progressive disclosure."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillReader,
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


def test_main_path_sanitizes_traversal_skill_name(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    """A ``../``-laden ``skill_name`` (raw LLM output) must not escape the agent dir.

    CWE-22 regression guard: prior to sanitization, ``skill_name`` was
    concatenated straight into the path, so a sufficiently long ``../``
    prefix resolved outside ``root.agents_dir()`` entirely. ``main_path``
    is a pure resolver (no frontmatter involved, no IO), matching how the
    traversal was originally measured.
    """
    traversal_name = "../" * 8 + "tmp/pwned"

    path = writer.main_path("agent_x", traversal_name)

    assert path.resolve().is_relative_to(root.agents_dir().resolve())
    assert path.name == "SKILL.md"
    assert "/" not in path.parent.name
    assert path.parent.parent == root.agents_dir() / "agent_x" / "skills"


_BOUNDARY_RAW_NAMES = [
    "..",
    "../",
    "/../",
    ".",
    "./",
    "!!!",  # sanitizes to empty -> fallback
    "a" * 200,  # truncation
    "修复 Django 自动重载问题",  # CJK + space
    "../" * 8 + "tmp/pwned",
]


@pytest.mark.parametrize("raw_name", _BOUNDARY_RAW_NAMES)
async def test_presanitized_name_identical_to_directory_segment(
    root: MemoryRoot, writer: AgentSkillWriter, raw_name: str
) -> None:
    """Mirrors ``extract_agent_skill._persist_skill``: sanitize
    ``skill_name`` once, up front, then use that same sanitized string for
    both the frontmatter ``name`` field and the writer's ``skill_name``
    argument. Covers the boundary family that previously slipped through
    the sanitizer as a fixpoint (``".."`` alone, or with a leading/trailing
    separator that strips down to it; ``"."`` likewise) in addition to the
    empty/truncation/CJK/traversal cases already covered.

    For each input: the sanitized name is a single path component
    (contains no separator), is never ``""`` / ``"."`` / ``".."``,
    constructing ``AgentSkillFrontmatter`` with it succeeds, and
    ``frontmatter.name`` is byte-identical (an identity, not merely
    idempotent-if-resanitized) to the directory segment actually written.
    """
    sanitized_name = AgentSkillFrontmatter.sanitize_skill_name(raw_name)

    assert "/" not in sanitized_name
    assert "\\" not in sanitized_name
    assert sanitized_name not in ("", ".", "..")

    fm = _make_fm(name=sanitized_name, id=f"agent_x_{sanitized_name}")

    path = await writer.write_main("agent_x", sanitized_name, frontmatter=fm, body="b")

    dir_derived_name = path.parent.name.removeprefix(
        AgentSkillFrontmatter.SKILL_DIR_PREFIX
    )
    assert fm.name == dir_derived_name


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


@pytest.mark.parametrize(
    ("reference_name", "script_filename"),
    [
        pytest.param("../" * 6 + "etc/passwd", "../" * 6 + "evil.sh", id="traversal"),
        pytest.param("..", "..", id="dotdot_fixpoint"),
        pytest.param("", "", id="empty"),
        pytest.param("notes/../../x", "run/../../x.sh", id="embedded_separators"),
    ],
)
async def test_reference_and_script_segments_cannot_escape_the_skill_dir(
    root: MemoryRoot,
    writer: AgentSkillWriter,
    reference_name: str,
    script_filename: str,
) -> None:
    """These two segments are appended *after* ``skill_dir_name``.

    ``skill_dir_name`` only sanitizes the ``skill_<name>`` component, so it
    offers these no protection at all — they need their own pass through
    ``sanitize_dirname``. Nothing in ``src/`` calls them yet; they are
    covered now because they are public API whose inputs will come from the
    same untrusted place the skill name does once progressive disclosure is
    wired up, and because the traversal fix would otherwise read as
    repo-wide when it is not.
    """
    skill_dir = root.agents_dir() / "agent_x" / "skills" / "skill_alpha"

    ref = await writer.write_reference("agent_x", "alpha", reference_name, "x")
    script = await writer.write_script("agent_x", "alpha", script_filename, "x")

    for path in (ref, script):
        assert path.is_relative_to(skill_dir)
        assert ".." not in path.parts
        assert path.is_file()


async def test_reader_resolves_the_same_sanitized_reference_and_script_paths(
    root: MemoryRoot, writer: AgentSkillWriter
) -> None:
    """Reader and writer must sanitize every segment identically.

    Sanitizing only one side would silently split a write from its matching
    read — the write lands on the safe path, the read looks at the raw one
    and reports the file missing. This is the same reader/writer symmetry
    ``skill_dir_name`` maintains for the skill directory, extended to the
    two segments appended after it.
    """
    reader = AgentSkillReader(root)
    await writer.write_reference("agent_x", "alpha", "my notes!", "ref body")
    await writer.write_script("agent_x", "alpha", "run this.sh", "echo hi\n")

    assert await reader.read_reference("agent_x", "alpha", "my notes!") == "ref body"
    assert await reader.read_script("agent_x", "alpha", "run this.sh") == "echo hi"
