"""AgentSkill frontmatter — single SKILL.md inside a skill directory.

Path: ``agents/<scope_id>/skills/skill_<name>/SKILL.md`` (plus sibling
``references/*.md`` and ``scripts/*.<ext>`` files that are not part of
the frontmatter contract).

Skills are *named entities* rather than daily-log entries: the
LanceDB primary key is ``<owner_id>_<skill_name>`` (no date / seq).
Upserts replace the file wholesale; the cascade daemon recomputes the
``content`` index column by concatenating ``SKILL.md`` body with every
``references/*.md`` sibling.

Five directory-shape ClassVars pin the layout in one place so the
writer / reader pair reads off them — no duplicated string literals.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from everos.core.persistence.markdown import (
    AgentScopedFrontmatter,
    SkillPathMixin,
)


class AgentSkillFrontmatter(SkillPathMixin, AgentScopedFrontmatter):
    """Frontmatter for ``agents/<scope>/skills/skill_<name>/SKILL.md``."""

    SKILLS_CONTAINER_NAME: ClassVar[str] = "skills"
    SKILL_DIR_PREFIX: ClassVar[str] = "skill_"
    SKILL_MAIN_FILENAME: ClassVar[str] = "SKILL.md"
    SKILL_REFERENCES_DIR_NAME: ClassVar[str] = "references"
    SKILL_SCRIPTS_DIR_NAME: ClassVar[str] = "scripts"

    type: Literal["agent_skill"] = "agent_skill"

    name: str
    """Skill identifier — also the directory suffix
    (``skills/skill_<name>/``). Keep snake_case so it is filesystem-safe
    and ID-stable."""

    description: str
    """One-line summary surfaced at Tier-1 prompt injection. Short — the
    agent's startup-time scanner reads ``(name, description)`` for every
    skill, so the token budget is tight."""

    confidence: float
    """LLM-emitted confidence in the skill's correctness, 0.0–1.0."""

    maturity_score: float
    """LLM-emitted maturity score, 0.0–1.0. The retrieval-time threshold
    (``maturity_threshold``) lives in MemorizeConfig, not on this file."""

    source_case_ids: list[str] = []
    """AgentCase ids that fed into this skill's synthesis (lineage)."""

    cluster_id: str | None = None
    """Optional MemScene clustering tag; may be unset early on."""

    created_at: _dt.datetime | None = None
    updated_at: _dt.datetime | None = None
