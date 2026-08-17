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

from pydantic import field_validator

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
    (``skills/skill_<name>/``, sanitized via
    :meth:`SkillPathMixin.skill_dir_name`). Keep snake_case so it stays
    readable and ID-stable; the directory segment is sanitized regardless."""

    @field_validator("name")
    @classmethod
    def _reject_path_traversal(cls, value: str) -> str:
        """Catch a frontmatter ``name`` that bypassed the writer's sanitizer.

        The normal write path
        (``memory.strategies.extract_agent_skill._persist_skill``) sanitizes
        LLM-emitted ``skill_name`` via
        :meth:`SkillPathMixin.sanitize_skill_name` *before* constructing this
        model, so ``name`` is traversal-free by the time it gets here on that
        path — this validator should not normally fire for LLM output at
        all. It exists for the case that does bypass the writer: a
        hand-edited ``SKILL.md`` (or any other direct
        ``AgentSkillFrontmatter`` construction that skips pre-sanitization)
        whose ``name`` contains a path separator, or is exactly ``".."`` —
        raise loudly rather than silently relocating the skill on next
        write.

        The check is deliberately narrower than "contains ``..``": a
        sanitized name may legitimately contain a run of literal dots
        (``sanitize_dirname`` keeps ``.`` as a safe character, so
        ``"../" * 8 + "tmp/pwned"`` sanitizes to
        ``"................tmppwned"``, which still contains the substring
        ``".."`` many times over). With no path separator left, that string
        is one opaque filename component, not a ``..`` traversal segment —
        rejecting on substring containment would make this validator
        reject the sanitizer's own safe output.
        """
        if "/" in value or "\\" in value or value == "..":
            raise ValueError(
                f"skill name {value!r} must not contain path separators, "
                "and must not be exactly '..'"
            )
        return value

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
