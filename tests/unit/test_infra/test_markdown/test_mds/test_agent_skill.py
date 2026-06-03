"""Tests for :class:`AgentSkillFrontmatter` — the AgentSkill schema.

Lives under ``test_infra`` because :class:`AgentSkillFrontmatter` itself
lives under ``infra/.../mds`` (it carries business fields + the
directory-shape ClassVars). The schema-agnostic chassis tests live
under ``test_core/test_persistence/test_markdown/``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.infra.persistence.markdown import AgentSkillFrontmatter


def _kwargs(**overrides: object) -> dict[str, object]:
    """Minimal valid kwargs for AgentSkillFrontmatter."""
    base: dict[str, object] = {
        "id": "skill_contract_risk_scan",
        "agent_id": "agent_zhang_legal",
        "name": "contract_risk_scan",
        "description": "Scan a contract draft for risk clauses.",
        "confidence": 0.5,
        "maturity_score": 0.5,
    }
    base.update(overrides)
    return base


def test_skill_inherits_agent_scope() -> None:
    """Skills always live under ``agents/`` — track + SCOPE_DIR confirm."""
    assert AgentSkillFrontmatter.SCOPE_DIR == "agents"
    fm = AgentSkillFrontmatter(**_kwargs())  # type: ignore[arg-type]
    assert fm.track == "agent"
    assert fm.type == "agent_skill"


def test_skill_requires_name_and_description() -> None:
    """Tier-1 prompt injection demands both fields — schema enforces."""
    bad = _kwargs()
    del bad["name"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]

    bad = _kwargs()
    del bad["description"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]


def test_skill_requires_confidence_and_maturity_score() -> None:
    """LLM-emitted score fields are required (no default)."""
    bad = _kwargs()
    del bad["confidence"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]

    bad = _kwargs()
    del bad["maturity_score"]
    with pytest.raises(ValidationError):
        AgentSkillFrontmatter(**bad)  # type: ignore[arg-type]


def test_skill_optional_fields_default() -> None:
    """``source_case_ids`` defaults to empty list; ``cluster_id`` to None."""
    fm = AgentSkillFrontmatter(**_kwargs())  # type: ignore[arg-type]
    assert fm.source_case_ids == []
    assert fm.cluster_id is None


def test_skill_lineage_fields_round_trip() -> None:
    """``source_case_ids`` + ``cluster_id`` round-trip through model_dump."""
    fm = AgentSkillFrontmatter(
        **_kwargs(
            source_case_ids=["case_a", "case_b"],
            cluster_id="cl_x",
        ),  # type: ignore[arg-type]
    )
    dumped = fm.model_dump()
    assert dumped["source_case_ids"] == ["case_a", "case_b"]
    assert dumped["cluster_id"] == "cl_x"


def test_skill_extra_fields_still_allowed() -> None:
    """L2 system metadata (md_sha256 / last_indexed_at) rides along."""
    fm = AgentSkillFrontmatter(
        **_kwargs(
            md_sha256="deadbeef",
            last_indexed_at="2026-05-07T08:00:00Z",
        ),  # type: ignore[arg-type]
    )
    dumped = fm.model_dump()
    assert dumped["md_sha256"] == "deadbeef"
    assert dumped["last_indexed_at"] == "2026-05-07T08:00:00Z"


def test_skill_directory_shape_classvars() -> None:
    """Path-shape ClassVars pin the wiki layout for the writer/reader pair."""
    assert AgentSkillFrontmatter.SKILLS_CONTAINER_NAME == "skills"
    assert AgentSkillFrontmatter.SKILL_DIR_PREFIX == "skill_"
    assert AgentSkillFrontmatter.SKILL_MAIN_FILENAME == "SKILL.md"
    assert AgentSkillFrontmatter.SKILL_REFERENCES_DIR_NAME == "references"
    assert AgentSkillFrontmatter.SKILL_SCRIPTS_DIR_NAME == "scripts"
