"""AgentSkillReader — typed read for the AgentSkill directory layout.

Pairs with :class:`AgentSkillWriter`:

- :meth:`read_main` reads ``SKILL.md`` and returns the caller's
  :class:`AgentSkillFrontmatter` subclass instance + the Tier-2 body, so
  the caller never deals with raw dicts.
- :meth:`read_reference` / :meth:`read_script` are plain text reads;
  no frontmatter, no schema.

All three return ``None`` when the target is missing — readers do not
raise on absence, since "skill not yet created" is a normal state for
the upsert-style workflow. Callers that need to distinguish "missing"
from "empty body" check for ``None`` explicitly.

Path resolution mirrors :class:`AgentSkillWriter` and reads the same
ClassVars off :class:`AgentSkillFrontmatter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import anyio

from everos.core.persistence import MarkdownReader, MemoryRoot

from ..mds import AgentSkillFrontmatter

T = TypeVar("T", bound=AgentSkillFrontmatter)


class AgentSkillReader:
    """Single-skill reader for the directory + progressive-disclosure layout."""

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root

    # ── Public API ────────────────────────────────────────────────────────

    async def read_main(
        self,
        agent_id: str,
        skill_name: str,
        *,
        schema: type[T],
        app_id: str = "default",
        project_id: str = "default",
    ) -> tuple[T, str] | None:
        """Read ``SKILL.md`` and parse its frontmatter into ``schema``.

        Args:
            schema: Concrete :class:`AgentSkillFrontmatter` subclass. The
                frontmatter dict is validated against this schema via
                :meth:`pydantic.BaseModel.model_validate`; extra fields
                ride along (chassis sets ``extra="allow"``).

        Returns:
            ``(frontmatter, body)`` on success, ``None`` if the file
            does not exist. ``body`` is the raw text after the closing
            ``---``; the trailing newline added by :class:`AgentSkillWriter`
            is stripped to give the *logical* body back.
        """
        path = self._main_path(agent_id, skill_name, app_id, project_id)
        if not await anyio.Path(path).is_file():
            return None
        parsed = await MarkdownReader.read(path)
        frontmatter = schema.model_validate(parsed.frontmatter)
        body = parsed.body.rstrip("\n")
        return frontmatter, body

    async def read_reference(
        self,
        agent_id: str,
        skill_name: str,
        reference_name: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> str | None:
        """Read ``references/<reference_name>.md`` verbatim, ``None`` if absent."""
        path = self._reference_path(
            agent_id, skill_name, reference_name, app_id, project_id
        )
        apath = anyio.Path(path)
        if not await apath.is_file():
            return None
        text = await apath.read_text(encoding="utf-8")
        return text.rstrip("\n")

    async def read_script(
        self,
        agent_id: str,
        skill_name: str,
        script_filename: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> str | None:
        """Read ``scripts/<script_filename>`` verbatim, ``None`` if absent.

        Reading ≠ executing — this only returns the source text.
        Sandboxing / exec-policy decisions belong to the caller.
        """
        path = self._script_path(
            agent_id, skill_name, script_filename, app_id, project_id
        )
        apath = anyio.Path(path)
        if not await apath.is_file():
            return None
        text = await apath.read_text(encoding="utf-8")
        return text.rstrip("\n")

    # ── Internals — same shape as AgentSkillWriter ────────────────────────────

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
