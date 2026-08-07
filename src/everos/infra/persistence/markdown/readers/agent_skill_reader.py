"""AgentSkillReader — typed read for the AgentSkill directory layout.

Pairs with :class:`AgentSkillWriter`:

- :meth:`read_main` reads ``SKILL.md`` and returns the caller's
  :class:`AgentSkillFrontmatter` subclass instance + the Tier-2 body, so
  the caller never deals with raw dicts.
- :meth:`list_by_cluster` walks every ``skill_*/SKILL.md`` under an agent
  and returns ``(frontmatter, body)`` for the ones whose parsed
  ``cluster_id`` matches. This is the strong-consistency source of truth
  for cluster membership — LanceDB is cascade-lagged and must not be used
  for that check.
- :meth:`read_reference` / :meth:`read_script` are plain text reads;
  no frontmatter, no schema.

``read_main``, ``read_reference``, and ``read_script`` return ``None`` when
the target is missing — readers do not raise on absence, since "skill not
yet created" is a normal state for the upsert-style workflow. Callers that
need to distinguish "missing" from "empty body" check for ``None``
explicitly.

``reference_name`` / ``script_filename`` are appended after the skill
directory, so ``skill_dir_name`` does not cover them; both go through
:func:`sanitize_dirname` here exactly as :class:`AgentSkillWriter` does.
The two sides must agree on *every* segment — sanitizing one side only
would route a write and its matching read to different paths.

Path resolution mirrors :class:`AgentSkillWriter` and reads the same
ClassVars off :class:`AgentSkillFrontmatter`, including
:meth:`AgentSkillFrontmatter.skill_dir_name` for the traversal-safe
directory segment. ``read_main`` / ``read_reference`` / ``read_script``
take a caller-supplied ``skill_name`` and re-derive the path from it, so
the reader and writer must never diverge on how a ``skill_name`` maps to
a directory. ``list_by_cluster`` never derives a path at all: it reads
each globbed ``SKILL.md`` path directly and hands back the body it
already read, rather than recovering a name from the directory and
leaving the caller to re-derive a path from that name for a second read.
A caller that discarded the body and re-read by name would recreate
exactly the re-sanitization risk this method exists to avoid — a
directory whose suffix isn't itself a sanitizer fixpoint (e.g. one
containing a raw space) would silently miss on that second, name-based
read even though the first, path-based read found it just fine. Returning
the body is what makes "the reader never derives a path" a property of
the full ``list_by_cluster`` → caller flow, not just of the enumeration
step in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import anyio
from pydantic import ValidationError

from everos.core.observability.logging import get_logger
from everos.core.persistence import MarkdownReader, MemoryRoot, sanitize_dirname

from ..mds import AgentSkillFrontmatter

T = TypeVar("T", bound=AgentSkillFrontmatter)

logger = get_logger(__name__)


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
        return await self._read_path(path, schema=schema)

    async def list_by_cluster(
        self,
        agent_id: str,
        cluster_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[tuple[AgentSkillFrontmatter, str]]:
        """Enumerate this agent's ``SKILL.md`` files whose ``cluster_id`` matches.

        Walks ``skills/skill_*/SKILL.md`` under the agent's memory root and
        returns ``(frontmatter, body)`` for each match. Skills whose
        frontmatter has ``cluster_id is None`` (or a different cluster) are
        filtered out.

        This is the strong-consistency source of truth for "which skills
        belong to this cluster" — LanceDB is cascade-lagged and must not be
        used for existence checks. Each glob match is read directly by its
        already-resolved ``path`` (see :meth:`_read_path`), *not* by
        recovering a ``skill_name`` from the directory and calling
        :meth:`read_main` to re-derive the same path — the reader never
        derives a path at all on this route, so this enumeration cannot drop
        a skill whose on-disk directory suffix is not itself a sanitizer
        fixpoint (e.g. one written with a raw, unsanitized name containing a
        space). Returning the body here (rather than frontmatter alone) is
        load-bearing, not a convenience: a caller that discarded it and
        re-read by ``frontmatter.name`` would reintroduce the same
        name-based re-derivation this method exists to avoid, one call
        later.

        A ``SKILL.md`` whose frontmatter fails schema validation is logged
        and skipped, not propagated. Isolating it matters because the
        blast radius of propagating is the whole cluster, not the one
        file: :meth:`_read_path` validates the full
        :class:`AgentSkillFrontmatter` schema, so *any* constraint can
        raise — a hand-edited ``name``, but equally a field that a later
        schema revision made required and existing files therefore lack.
        A single bad file would otherwise abort the enumeration, starve
        ``extract_agent_skill`` of every existing skill in the cluster,
        and dead-letter that cluster's extraction on every subsequent
        run — the exact permanent-failure mode this md-first read path
        exists to eliminate.

        Args:
            agent_id: Owning agent.
            cluster_id: Cluster to filter on.
            app_id: App scope; defaults to ``"default"``.
            project_id: Project scope; defaults to ``"default"``.

        Returns:
            ``(frontmatter, body)`` pairs sorted by skill directory path.
            Empty if the agent has no skill directory yet, none match, or
            every candidate failed validation.
        """
        skills_dir = self._skills_root(agent_id, app_id, project_id)
        if not await anyio.Path(skills_dir).is_dir():
            return []
        pattern = (
            f"{AgentSkillFrontmatter.SKILL_DIR_PREFIX}*"
            f"/{AgentSkillFrontmatter.SKILL_MAIN_FILENAME}"
        )
        paths = await anyio.to_thread.run_sync(lambda: sorted(skills_dir.glob(pattern)))
        matches: list[tuple[AgentSkillFrontmatter, str]] = []
        for path in paths:
            try:
                parsed = await self._read_path(path, schema=AgentSkillFrontmatter)
            except ValidationError as exc:
                logger.warning(
                    "agent_skill.list_by_cluster.unparseable_skill_skipped",
                    path=str(path),
                    cluster_id=cluster_id,
                    error=str(exc),
                )
                continue
            if parsed is None:
                continue
            frontmatter, body = parsed
            if frontmatter.cluster_id == cluster_id:
                matches.append((frontmatter, body))
        return matches

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

    async def _read_path(self, path: Path, *, schema: type[T]) -> tuple[T, str] | None:
        """Read + parse an already-resolved ``SKILL.md`` path.

        Shared by :meth:`read_main` (path derived from a caller-supplied
        ``skill_name``) and :meth:`list_by_cluster` (path taken directly
        from a directory glob, never re-derived from a name).

        Raises:
            ValidationError: the file's frontmatter violates *schema*.
                Propagated to the caller — ``read_main`` asked for one
                specific skill and a corrupt answer is not a substitute,
                while ``list_by_cluster`` catches it per file so one bad
                file cannot starve the rest of the cluster.
        """
        if not await anyio.Path(path).is_file():
            return None
        parsed = await MarkdownReader.read(path)
        frontmatter = schema.model_validate(parsed.frontmatter)
        body = parsed.body.rstrip("\n")
        return frontmatter, body

    def _skills_root(self, agent_id: str, app_id: str, project_id: str) -> Path:
        return (
            self._root.agents_dir(app_id, project_id)
            / agent_id
            / AgentSkillFrontmatter.SKILLS_CONTAINER_NAME
        )

    def _skill_dir(
        self, agent_id: str, skill_name: str, app_id: str, project_id: str
    ) -> Path:
        return self._skills_root(
            agent_id, app_id, project_id
        ) / AgentSkillFrontmatter.skill_dir_name(skill_name)

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
            / f"{sanitize_dirname(reference_name, 'reference')}.md"
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
            / sanitize_dirname(script_filename, "script")
        )
