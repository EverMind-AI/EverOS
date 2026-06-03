"""ProfileWriter — upsert a single-file, fixed-name profile markdown.

Profile storage is **single-file rewrite** (the third storage strategy
in the EverOS Markdown First spec). Each profile lives at a fixed
filename under the agent or user directory::

    users/<user_id>/user.md          ← user profile
    users/<user_id>/behaviors.md     ← user behaviour patterns
    agents/<agent_id>/agent.md       ← agent playbook
    agents/<agent_id>/soul.md        ← agent identity / values
    agents/<agent_id>/tools.md       ← agent tool declarations

Compared with :class:`SkillWriter` (directory + progressive disclosure)
and :class:`BaseDailyWriter` (per-date append + entry markers), the
profile writer is the simplest of the three:

- **Upsert, not append.** Each ``write`` overwrites the file in full.
- **Fixed path.** Caller passes ``scope_id`` only — no ``name``
  parameter; the filename is fixed by the schema's
  ``PROFILE_FILENAME`` ClassVar.
- **No business hooks.** No frontmatter merging, no entry-id
  generation. The caller hands in a fully-built schema instance.

The schema must declare two ClassVars:

- ``SCOPE_DIR`` (``"users"`` / ``"agents"``) — inherited from
  :class:`UserScopedFrontmatter` / :class:`AgentScopedFrontmatter`.
- ``PROFILE_FILENAME`` (``"user.md"`` / ``"agent.md"`` / …) —
  declared on the concrete profile schema itself.

There is no ``ProfileFrontmatter`` base class: profile schemas are
duck-typed via the two ClassVars. Subclasses inherit the scope mixin
and add ``PROFILE_FILENAME`` plus their business fields directly.
"""

from __future__ import annotations

from pathlib import Path

from everos.core.persistence import BaseFrontmatter, MarkdownWriter, MemoryRoot


class ProfileWriter:
    """Atomic writer for the single-file profile layout."""

    def __init__(
        self,
        root: MemoryRoot,
        *,
        writer: MarkdownWriter | None = None,
    ) -> None:
        self._root = root
        self._writer = writer or MarkdownWriter(root)

    # ── Public API ────────────────────────────────────────────────────────

    async def write(
        self,
        scope_id: str,
        *,
        frontmatter: BaseFrontmatter,
        body: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Upsert ``<app>/<project>/<scope>/<scope_id>/<PROFILE_FILENAME>``.

        Args:
            scope_id: ``user_id`` or ``agent_id`` (must match the
                schema's scope mixin).
            frontmatter: Fully-built schema instance — its ``model_dump``
                lands as the YAML head, including extra fields.
            body: Profile body text. Trailing newline is normalised.
            app_id: App scope segment (defaults to the ``"default"`` space).
            project_id: Project scope segment (defaults to ``"default"``).

        Returns:
            Absolute path of the written profile file.
        """
        path = self._resolve_path(scope_id, type(frontmatter), app_id, project_id)
        head_meta = frontmatter.model_dump(exclude_none=False)
        return await self._writer.write_markdown(
            path,
            frontmatter=head_meta,
            body=_ensure_trailing_newline(body),
        )

    def path_for(
        self,
        scope_id: str,
        *,
        schema: type[BaseFrontmatter],
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Return the profile path (no IO check)."""
        return self._resolve_path(scope_id, schema, app_id, project_id)

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_path(
        self,
        scope_id: str,
        schema: type[BaseFrontmatter],
        app_id: str,
        project_id: str,
    ) -> Path:
        scope_dir = getattr(schema, "SCOPE_DIR", "")
        filename = getattr(schema, "PROFILE_FILENAME", None)
        if not scope_dir:
            raise TypeError(
                f"{schema.__name__} missing ``SCOPE_DIR`` ClassVar — "
                "must inherit a scope mixin (UserScopedFrontmatter / "
                "AgentScopedFrontmatter)."
            )
        if not filename:
            raise TypeError(f"{schema.__name__} missing ``PROFILE_FILENAME`` ClassVar.")
        # SCOPE_DIR names the matching MemoryRoot method (<app>/<project> prefix).
        scope_root = getattr(self._root, f"{scope_dir}_dir")(app_id, project_id)
        return scope_root / scope_id / filename


def _ensure_trailing_newline(text: str) -> str:
    """End the body with exactly one newline (POSIX text-file convention)."""
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"
