"""ProfileReader — typed read for the single-file profile layout.

Pairs with :class:`ProfileWriter`. The schema (concrete profile
frontmatter class) is supplied per call; the reader pulls
``SCOPE_DIR`` + ``PROFILE_FILENAME`` ClassVars off it to build the
path, then ``MarkdownReader.read`` + ``schema.model_validate`` give
back a typed frontmatter instance plus the body string.

Returns ``None`` when the profile file does not exist — "not yet
written" is a normal early state for the upsert-style workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import anyio

from everos.core.persistence import BaseFrontmatter, MarkdownReader, MemoryRoot

T = TypeVar("T", bound=BaseFrontmatter)


class ProfileReader:
    """Typed read for fixed-name profile markdown files."""

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root

    # ── Public API ────────────────────────────────────────────────────────

    async def read(
        self,
        scope_id: str,
        *,
        schema: type[T],
        app_id: str = "default",
        project_id: str = "default",
    ) -> tuple[T, str] | None:
        """Read the profile file and parse its frontmatter into ``schema``.

        Args:
            scope_id: ``user_id`` or ``agent_id`` (must match the
                schema's scope mixin).
            schema: Concrete profile frontmatter class — must declare
                ``SCOPE_DIR`` (via scope mixin) and ``PROFILE_FILENAME``.
            app_id: App scope segment (defaults to the ``"default"`` space).
            project_id: Project scope segment (defaults to ``"default"``).

        Returns:
            ``(frontmatter, body)`` on success; ``None`` if the file is
            missing. ``body`` is the raw text after the closing ``---``
            with the writer-added trailing newline stripped.
        """
        path = self._resolve_path(scope_id, schema, app_id, project_id)
        if not await anyio.Path(path).is_file():
            return None
        parsed = await MarkdownReader.read(path)
        frontmatter = schema.model_validate(parsed.frontmatter)
        body = parsed.body.rstrip("\n")
        return frontmatter, body

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

    # ── Internals — same shape as ProfileWriter ───────────────────────────

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
