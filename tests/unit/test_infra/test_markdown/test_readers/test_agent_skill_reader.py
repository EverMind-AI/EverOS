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

    names = sorted(fm.name for fm, _body in results)
    assert names == ["drain_queue", "revive_replica"]
    assert all(body == "b" for _fm, body in results)


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

    results = await reader.list_by_cluster("a1", "cl1")
    assert [fm.name for fm, _body in results] == ["assigned"]
    assert await reader.list_by_cluster("a1", "cl_missing") == []


async def test_list_by_cluster_missing_dir_returns_empty(
    reader: AgentSkillReader,
) -> None:
    """New agent with no skill dir yet — returns [] without raising."""
    assert await reader.list_by_cluster("a_new", "cl1") == []


async def test_list_by_cluster_finds_skill_whose_directory_suffix_has_a_space(
    root: MemoryRoot, reader: AgentSkillReader
) -> None:
    """Regression guard: before the fix, ``list_by_cluster`` recovered
    ``skill_name`` from the directory suffix and called ``read_main``,
    which re-derives (and re-sanitizes) the path from that name. A
    directory whose suffix is not itself already a sanitizer fixpoint —
    e.g. ``skill_My Skill`` (a raw space, never passed through
    ``sanitize_dirname``) — re-derived to ``skill_My_Skill``, a path that
    does not exist, so the skill was silently dropped from the result.
    ``list_by_cluster`` is documented as the strong-consistency existence
    check for cluster membership, so a dropped skill here would make the
    LLM emit ``add()`` for a skill that already exists, duplicating it at
    the sanitized path and orphaning the original.

    Writes the ``SKILL.md`` directly to the filesystem (bypassing both
    ``AgentSkillWriter`` and ``_persist_skill``) to reproduce a directory
    that was never sanitized in the first place — the scenario the fix
    (reading the globbed path directly, never re-deriving it) must cover
    regardless of how such a directory came to exist. Asserts the body
    too, not just enumeration: a fix that stopped at "the frontmatter is
    found" but still forced a caller to re-read by name (re-derive, and
    re-sanitize, the same path) would have moved the drop one layer
    downstream rather than closing it — see
    ``test_extract_agent_skill.test_select_existing_skills_finds_skill_whose_directory_suffix_has_a_space``
    for the end-to-end version of this same property.
    """
    skill_dir = root.agents_dir() / "a1" / "skills" / "skill_My Skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "id: a1_My Skill\n"
        "type: agent_skill\n"
        "agent_id: a1\n"
        "track: agent\n"
        "name: My Skill\n"
        "description: d\n"
        "confidence: 0.5\n"
        "maturity_score: 0.5\n"
        "cluster_id: cl1\n"
        "---\n"
        "The real skill body.\n",
        encoding="utf-8",
    )

    results = await reader.list_by_cluster("a1", "cl1")

    assert len(results) == 1
    fm, body = results[0]
    assert fm.name == "My Skill"
    assert body == "The real skill body."


@pytest.mark.parametrize(
    "frontmatter_lines",
    [
        # Any schema constraint can fail, not just the traversal validator —
        # a field a later schema revision makes required is the case existing
        # files on disk would hit at upgrade time, all at once.
        pytest.param("name: broken\ncluster_id: cl1\n", id="missing_required_field"),
        # The traversal validator, whose whole stated purpose is catching a
        # hand-edited name — i.e. a file that reaches exactly this route.
        pytest.param(
            "name: ..\ndescription: d\nconfidence: 0.5\n"
            "maturity_score: 0.5\ncluster_id: cl1\n",
            id="traversal_name",
        ),
    ],
)
async def test_list_by_cluster_skips_unparseable_skill(
    root: MemoryRoot,
    reader: AgentSkillReader,
    frontmatter_lines: str,
) -> None:
    """One unparseable ``SKILL.md`` must not starve the whole cluster.

    ``_read_path`` validates the full :class:`AgentSkillFrontmatter` schema,
    so a single malformed file used to abort the entire enumeration — and
    the enumeration is what feeds ``extract_agent_skill`` its existing
    skills. Propagating would leave that strategy raising on every run for
    the whole cluster, i.e. permanently dead-lettered: the exact failure
    mode the md-first read path was introduced to eliminate. The good files
    on either side of the bad one pin that the skip is per-file rather than
    "stop at the first error" — sorted glob order puts ``skill_bad`` between
    them, so a fix that merely stopped raising would still lose ``zzz``.
    """
    skills = root.agents_dir() / "a1" / "skills"

    def _write(dirname: str, body_frontmatter: str, body: str) -> None:
        d = skills / dirname
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\ntype: agent_skill\nagent_id: a1\ntrack: agent\n"
            f"id: a1_{dirname}\n{body_frontmatter}---\n{body}\n",
            encoding="utf-8",
        )

    good = "description: d\nconfidence: 0.5\nmaturity_score: 0.5\ncluster_id: cl1\n"
    _write("skill_aaa", f"name: aaa\n{good}", "body aaa")
    _write("skill_bad", frontmatter_lines, "body bad")
    _write("skill_zzz", f"name: zzz\n{good}", "body zzz")

    results = await reader.list_by_cluster("a1", "cl1")

    assert [fm.name for fm, _ in results] == ["aaa", "zzz"]
    assert [body for _, body in results] == ["body aaa", "body zzz"]


async def test_read_main_propagates_validation_error(
    root: MemoryRoot, reader: AgentSkillReader
) -> None:
    """``read_main`` keeps raising — the skip is scoped to enumeration.

    A caller naming one specific skill gets an error rather than ``None``:
    ``None`` means "not created yet", a normal state this reader's callers
    branch on, and silently reusing it for "exists but is corrupt" would let
    an upsert overwrite the damaged file instead of surfacing it.
    """
    skill_dir = root.agents_dir() / "a1" / "skills" / "skill_aaa"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ntype: agent_skill\nagent_id: a1\ntrack: agent\n"
        "id: a1_aaa\nname: aaa\n---\nbody\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        await reader.read_main("a1", "aaa", schema=AgentSkillFrontmatter)


async def test_read_main_rederivation_from_raw_name_is_idempotent_safe(
    writer: AgentSkillWriter, reader: AgentSkillReader
) -> None:
    """``read_main`` re-derives a path from a caller-supplied name; that
    re-derivation must land on the same file a direct write produced, even
    when the name it's given is raw and unsanitized.

    No production caller currently re-derives a path from
    ``list_by_cluster``'s output — it returns each skill's body directly
    (see ``AgentSkillReader.list_by_cluster``'s docstring), so
    ``extract_agent_skill`` never calls ``read_main`` in that flow anymore.
    ``read_main`` remains a general single-skill lookup on the reader's
    public API, so this test pins its re-derivation as a property of the
    method itself: writing via a *raw*, unsanitized name directly through
    the writer, then reading back via that same raw name, must resolve to
    the same file (idempotent-safe), for any future caller that does pass
    a raw name.
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

    out = await reader.read_main("a1", space_name, schema=AgentSkillFrontmatter)
    assert out is not None
    fm_out, body = out
    assert fm_out.name == space_name
    assert body == "The fix body."
