"""AgentSkillWriter — upsert skill main file + reference / script attachments.

Skill storage is **directory + progressive disclosure** (wiki "Memory
Types Markdown Format" v4): each skill lives under
``agents/<agent_id>/skills/skill_<name>/`` with a ``SKILL.md`` main
file plus ``references/*.md`` and ``scripts/*.<ext>`` siblings.

This writer is intentionally distinct from :class:`BaseDailyWriter`:

- **Upsert, not append.** Each ``write_*`` call overwrites the target
  file in full. Skills don't accumulate entry markers — the body of
  ``SKILL.md`` is the latest revision; references / scripts are
  individually replaceable files.
- **Single-skill API.** The chassis is *not* responsible for bulk
  enumeration (Tier-1 prompt scanning is a sqlite/lancedb concern,
  not a markdown-walk concern). One skill in, one skill out.
- **No counters / hooks.** No frontmatter merging, no entry-id
  generation, no _frontmatter_updates hook — the caller hands in a
  fully-built :class:`AgentSkillFrontmatter` subclass instance and the body
  string; the writer atomically replaces the file.

Path resolution comes from :class:`MemoryRoot` + the ClassVars on
:class:`AgentSkillFrontmatter` (``SKILLS_CONTAINER_NAME`` /
``SKILL_DIR_PREFIX`` / etc.). The writer + reader pair is the single
addressing API for skills.
"""

from __future__ import annotations

from pathlib import Path

from everos.core.persistence import MarkdownWriter, MemoryRoot

from ..mds import AgentSkillFrontmatter


class AgentSkillWriter:
    """Atomic writer for the AgentSkill directory layout.

    Holds a :class:`MarkdownWriter` for the SKILL.md path (frontmatter +
    body) and a thin atomic-write helper for plain-text reference /
    script files (no frontmatter).
    """

    def __init__(
        self,
        root: MemoryRoot,
        *,
        writer: MarkdownWriter | None = None,
    ) -> None:
        self._root = root
        self._writer = writer or MarkdownWriter(root)

    # ── Public API ────────────────────────────────────────────────────────

    async def write_main(
        self,
        agent_id: str,
        skill_name: str,
        *,
        frontmatter: AgentSkillFrontmatter,
        body: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Upsert ``skills/skill_<name>/SKILL.md``.

        The file is replaced in full: ``frontmatter`` becomes the new
        YAML head, ``body`` becomes the new body. Any prior content
        (including manual human edits) is overwritten. The atomic
        rename keeps readers from ever seeing a torn write.

        Args:
            agent_id: Owning agent.
            skill_name: Unprefixed identifier (``"contract_risk_scan"``,
                not ``"skill_contract_risk_scan"``).
            frontmatter: Fully-built schema instance — its ``model_dump``
                is what lands in the YAML head, including extra fields.
            body: Tier-2 body text. Trailing newline is normalised.

        Returns:
            Absolute path of the written ``SKILL.md``.
        """
        path = self._main_path(agent_id, skill_name, app_id, project_id)
        head_meta = frontmatter.model_dump(exclude_none=False)
        return await self._writer.write_markdown(
            path,
            frontmatter=head_meta,
            body=_ensure_trailing_newline(body),
        )

    async def write_reference(
        self,
        agent_id: str,
        skill_name: str,
        reference_name: str,
        content: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Upsert ``skills/skill_<name>/references/<reference_name>.md``.

        Reference files are plain markdown — no frontmatter, no entry
        markers. Content is written verbatim (with a normalised
        trailing newline).

        Args:
            reference_name: Filename stem (no ``.md`` extension).
        """
        path = self._reference_path(
            agent_id, skill_name, reference_name, app_id, project_id
        )
        return await self._writer.write(path, _ensure_trailing_newline(content))

    async def write_script(
        self,
        agent_id: str,
        skill_name: str,
        script_filename: str,
        content: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Upsert ``skills/skill_<name>/scripts/<script_filename>``.

        Script files are written verbatim — caller supplies the *full*
        filename (including the extension; ``redline.py`` /
        ``redline.sh`` / etc.) since scripts may be in any language.
        Cascade does not index this directory.
        """
        path = self._script_path(
            agent_id, skill_name, script_filename, app_id, project_id
        )
        return await self._writer.write(path, _ensure_trailing_newline(content))

    # ── Path API (callers that need to echo paths in responses) ──────────

    def main_path(
        self,
        agent_id: str,
        skill_name: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Return ``skills/skill_<name>/SKILL.md`` (does not check existence)."""
        return self._main_path(agent_id, skill_name, app_id, project_id)

    # ── Internals — path resolution from AgentSkillFrontmatter ClassVars ──────

    def _skill_dir(
        self, agent_id: str, skill_name: str, app_id: str, project_id: str
    ) -> Path:
        return (
            self._root.agents_dir(app_id, project_id)
            / agent_id
            / AgentSkillFrontmatter.SKILLS_CONTAINER_NAME
            / f"{AgentSkillFrontmatter.SKILL_DIR_PREFIX}{skill_name}"
        )

    def _main_path(
        self, agent_id: str, skill_name: str, app_id: str, project_id: str
    ) -> Path:
        return (
            self._skill_dir(agent_id, skill_name, app_id, project_id)
            / AgentSkillFrontmatter.SKILL_MAIN_FILENAME
        )

    def _reference_path(
        self,
        agent_id: str,
        skill_name: str,
        reference_name: str,
        app_id: str,
        project_id: str,
    ) -> Path:
        return (
            self._skill_dir(agent_id, skill_name, app_id, project_id)
            / AgentSkillFrontmatter.SKILL_REFERENCES_DIR_NAME
            / f"{reference_name}.md"
        )

    def _script_path(
        self,
        agent_id: str,
        skill_name: str,
        script_filename: str,
        app_id: str,
        project_id: str,
    ) -> Path:
        return (
            self._skill_dir(agent_id, skill_name, app_id, project_id)
            / AgentSkillFrontmatter.SKILL_SCRIPTS_DIR_NAME
            / script_filename
        )


def _ensure_trailing_newline(text: str) -> str:
    """End the body with exactly one newline (POSIX text-file convention)."""
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"
