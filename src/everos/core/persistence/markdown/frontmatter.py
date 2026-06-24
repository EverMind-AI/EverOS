"""Frontmatter — YAML block parse / dump + L1 schema chassis.

Frontmatter is the leading ``---``-delimited YAML block at the top of
a markdown document::

    ---
    title: Hello
    tags: [a, b]
    ---
    # Body starts here

Two complementary surfaces live here:

1. :func:`parse_frontmatter` / :func:`dump_frontmatter` — schema-free
   YAML helpers (``yaml.safe_load`` / ``yaml.safe_dump``,
   ``sort_keys=False`` so caller-controlled key order is preserved).

2. The L1 chassis classes — :class:`BaseFrontmatter`,
   :class:`UserScopedFrontmatter`, :class:`AgentScopedFrontmatter` —
   which fix the *absolute-readonly* fields (``id`` / ``type`` /
   ``schema_version``) plus scope (``user_id`` / ``agent_id`` +
   ``track``). Every business frontmatter schema in
   ``infra/persistence/markdown/mds/`` subclasses one of these.

Concrete business schemas (``UserMemcellDailyFrontmatter``,
``SkillFrontmatter``, …) live in ``infra``; they add per-record
business fields plus the path-resolution metadata daily-log writers
need (``ENTRY_ID_PREFIX`` / ``DIR_NAME`` / ``FILE_PREFIX``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict

# ── YAML helpers ────────────────────────────────────────────────────────

_DELIM = "---"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a leading ``---\\n...\\n---\\n`` YAML block.

    Returns:
        (meta, remainder): ``meta`` is the parsed YAML mapping (empty dict
        if no frontmatter present, malformed, or non-mapping). ``remainder``
        is everything after the closing delimiter line — including the body's
        leading content as-is.

    Notes:
        - If the document does not start with ``---``, returns ``({}, text)``
          unchanged.
        - If a closing ``---`` line is not found, returns ``({}, text)``.
        - If the YAML block is empty (``---\\n---\\n``), returns
          ``({}, remainder)``.
        - If the parsed YAML is not a mapping (e.g. a scalar list), returns
          ``({}, text)`` — frontmatter must be a mapping.
    """
    if not text.startswith(_DELIM):
        return {}, text

    # Skip the opening "---" and the newline that must follow it.
    rest = text[len(_DELIM) :]
    if rest.startswith("\r\n"):
        rest = rest[2:]
    elif rest.startswith("\n"):
        rest = rest[1:]
    else:
        # Opening "---" not followed by a newline → not a valid frontmatter.
        return {}, text

    closing_idx = _find_closing_delim(rest)
    if closing_idx is None:
        return {}, text

    yaml_block = rest[:closing_idx]
    remainder = rest[closing_idx + len(_DELIM) :]
    # Drop the newline that follows the closing delimiter, if any.
    if remainder.startswith("\r\n"):
        remainder = remainder[2:]
    elif remainder.startswith("\n"):
        remainder = remainder[1:]

    parsed: Any = yaml.safe_load(yaml_block) if yaml_block.strip() else {}
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return {}, text
    return parsed, remainder


def dump_frontmatter(meta: Mapping[str, Any]) -> str:
    """Render a mapping as a ``---\\n<yaml>\\n---\\n`` block.

    An empty mapping yields the empty string (no delimiters). The YAML
    payload preserves caller-supplied key order (``sort_keys=False``).
    """
    if not meta:
        return ""
    yaml_block = yaml.safe_dump(
        dict(meta),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return f"{_DELIM}\n{yaml_block}{_DELIM}\n"


def _find_closing_delim(text: str) -> int | None:
    """Find the offset of a line that is exactly ``---``.

    A "line" is text between two newlines (or string boundaries).
    Returns the offset of the first character of the matching line, or
    ``None`` if no such line exists.
    """
    pos = 0
    while pos < len(text):
        nl = text.find("\n", pos)
        line = text[pos:nl] if nl != -1 else text[pos:]
        if line.rstrip("\r") == _DELIM:
            return pos
        if nl == -1:
            return None
        pos = nl + 1
    return None


# ── L1 schema chassis ───────────────────────────────────────────────────


class BaseFrontmatter(BaseModel):
    """L1 fields every markdown frontmatter must carry.

    These match the *absolute-readonly* tier in the EverOS Markdown First
    spec — they identify the record across markdown ↔ LanceDB and must
    never be rewritten by a human edit.

    Subclasses add scope (``UserScopedFrontmatter`` /
    ``AgentScopedFrontmatter``) plus per-record business fields.
    """

    SCOPE_DIR: ClassVar[str] = ""
    """Top-level directory under the memory-root that holds this kind.

    Scope mixins set this to ``"users"`` / ``"agents"``. Scope-agnostic
    schemas (rare) leave it empty; consumers that need to resolve a path
    (writers, layout reverse-lookup) must reject schemas with empty
    ``SCOPE_DIR``.
    """

    id: str
    type: str
    schema_version: int = 1

    # Permit additional fields so L2 system-managed metadata
    # (``md_sha256``, ``last_indexed_at``, ``lsn``, …) can ride along on
    # the same model without forcing every subclass to redeclare them.
    model_config = ConfigDict(extra="allow")

    @classmethod
    def path_glob(cls) -> str:
        """Return an ``fnmatch``-style glob (relative to memory-root)
        covering every markdown file this schema describes.

        Used by the cascade kind registry — the scanner walks every kind's
        ``path_glob()`` to enumerate eligible files without hard-coding
        path patterns in cascade. The schema is the single source of truth
        for both the writer's path resolution and the scanner's enumeration.

        Subclasses must override — typically by mixing in
        :class:`DailyLogPathMixin` or :class:`SkillPathMixin` *before* the
        scope mixin in the MRO so this abstract version is shadowed.
        """
        raise NotImplementedError(
            f"{cls.__name__} must declare path_glob() "
            f"(mix in DailyLogPathMixin / SkillPathMixin, or override directly)"
        )


class DailyLogPathMixin:
    """Path strategy for daily-log files.

    Files live at ``<SCOPE_DIR>/<scope_id>/<DIR_NAME>/<FILE_PREFIX>-<YYYY-MM-DD>.md``.
    Subclasses must inherit a scope mixin (``UserScopedFrontmatter`` /
    ``AgentScopedFrontmatter``) supplying ``SCOPE_DIR``, and must declare
    their own ``DIR_NAME`` / ``FILE_PREFIX`` ClassVars.

    Place **this mixin first** so Python's MRO resolves ``path_glob()`` to
    the mixin's concrete implementation rather than
    :meth:`BaseFrontmatter.path_glob`'s ``NotImplementedError`` stub::

        class EpisodeDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
            DIR_NAME: ClassVar[str] = "episodes"
            FILE_PREFIX: ClassVar[str] = "episode"
            ...
    """

    DIR_NAME: ClassVar[str]
    FILE_PREFIX: ClassVar[str]
    SCOPE_DIR: ClassVar[str]

    @classmethod
    def path_glob(cls) -> str:
        # Leading ``*/*/`` matches the <app>/<project> scope prefix that
        # precedes every user-visible dir; the scanner's ``root.glob`` is
        # anchored at root, so the prefix is mandatory (without it nothing
        # matches), and the watcher's right-anchored ``PurePosixPath.match``
        # agrees on the same shape.
        return f"*/*/{cls.SCOPE_DIR}/*/{cls.DIR_NAME}/{cls.FILE_PREFIX}-*.md"


class SkillPathMixin:
    """Path strategy for skill-directory files.

    Each skill lives at ``<SCOPE_DIR>/<scope_id>/<SKILLS_CONTAINER_NAME>/
    <SKILL_DIR_PREFIX><skill_name>/<SKILL_MAIN_FILENAME>``. The glob covers
    every skill's main file; sibling ``references/*.md`` and ``scripts/*``
    are excluded (they ride alongside the main file and the cascade
    daemon rebuilds the index column by concatenation, see
    :class:`AgentSkillFrontmatter`'s docstring).

    Place **this mixin first** so MRO resolves ``path_glob()`` here::

        class AgentSkillFrontmatter(SkillPathMixin, AgentScopedFrontmatter):
            SKILLS_CONTAINER_NAME: ClassVar[str] = "skills"
            SKILL_DIR_PREFIX: ClassVar[str] = "skill_"
            SKILL_MAIN_FILENAME: ClassVar[str] = "SKILL.md"
            ...
    """

    SKILLS_CONTAINER_NAME: ClassVar[str]
    SKILL_DIR_PREFIX: ClassVar[str]
    SKILL_MAIN_FILENAME: ClassVar[str]
    SCOPE_DIR: ClassVar[str]

    @classmethod
    def path_glob(cls) -> str:
        # Leading ``*/*/`` matches the <app>/<project> scope prefix.
        return (
            f"*/*/{cls.SCOPE_DIR}/*/{cls.SKILLS_CONTAINER_NAME}/"
            f"{cls.SKILL_DIR_PREFIX}*/{cls.SKILL_MAIN_FILENAME}"
        )


class ProfilePathMixin:
    """Path strategy for single-file profile markdown.

    Profiles live at ``<SCOPE_DIR>/<scope_id>/<PROFILE_FILENAME>`` —
    one fixed-name file directly under the scope's owner directory, no
    intermediate ``<dir>/`` segment (unlike daily-logs) and no per-name
    subdir (unlike skills). Subclasses must inherit a scope mixin
    (``UserScopedFrontmatter`` / ``AgentScopedFrontmatter``) supplying
    ``SCOPE_DIR`` and declare their own ``PROFILE_FILENAME``.

    Place **this mixin first** so MRO resolves ``path_glob()`` here::

        class UserProfileFrontmatter(ProfilePathMixin, UserScopedFrontmatter):
            PROFILE_FILENAME: ClassVar[str] = "user.md"
            ...
    """

    PROFILE_FILENAME: ClassVar[str]
    SCOPE_DIR: ClassVar[str]

    @classmethod
    def path_glob(cls) -> str:
        # Leading ``*/*/`` matches the <app>/<project> scope prefix.
        return f"*/*/{cls.SCOPE_DIR}/*/{cls.PROFILE_FILENAME}"


class UserScopedFrontmatter(BaseFrontmatter):
    """Records that belong to a single user (track = ``user``).

    The frontmatter only carries the *file-level* scope (``user_id``,
    which the path itself already expresses); business attributes like
    ``group_id`` live inside each entry's structured body — see
    :class:`StructuredEntry` in :mod:`.entries`.
    """

    SCOPE_DIR: ClassVar[str] = "users"

    user_id: str
    track: Literal["user"] = "user"


class AgentScopedFrontmatter(BaseFrontmatter):
    """Records that belong to a single agent (track = ``agent``).

    Same scope-vs-business split as :class:`UserScopedFrontmatter`:
    ``agent_id`` is the file-level scope; ``group_id`` etc. ride on
    each entry, not on the file frontmatter.
    """

    SCOPE_DIR: ClassVar[str] = "agents"

    agent_id: str
    track: Literal["agent"] = "agent"


class KnowledgeScopedMixin:
    """Records in the knowledge scope (no owner_id).

    Unlike user/agent scopes, knowledge records are not owned by an
    individual entity. The scope directory is ``knowledge/``.
    """

    SCOPE_DIR: ClassVar[str] = "knowledge"


class KnowledgeDocumentPathMixin:
    """Path strategy for ``knowledge/{category}/{doc_title}/index.md``.

    Place this mixin first so MRO resolves ``path_glob()`` here::

        class KnowledgeDocumentFrontmatter(
            KnowledgeDocumentPathMixin, KnowledgeScopedMixin, BaseFrontmatter
        ): ...
    """

    SCOPE_DIR: ClassVar[str]

    @classmethod
    def path_glob(cls) -> str:
        return f"*/*/{cls.SCOPE_DIR}/*/*/index.md"


class KnowledgeTopicPathMixin:
    """Path strategy for ``knowledge/{category}/{doc_title}/<n>_<name>.md``.

    Place this mixin first so MRO resolves ``path_glob()`` here::

        class KnowledgeTopicFrontmatter(
            KnowledgeTopicPathMixin, KnowledgeScopedMixin, BaseFrontmatter
        ): ...
    """

    SCOPE_DIR: ClassVar[str]

    @classmethod
    def path_glob(cls) -> str:
        return f"*/*/{cls.SCOPE_DIR}/*/*/[0-9]*.md"
