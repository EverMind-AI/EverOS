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
   :class:`UserScopedFrontmatter`, :class:`GroupScopedFrontmatter` — which
   fix the *absolute-readonly* fields (``id`` / ``type`` /
   ``schema_version``) plus scope (``user_id`` / ``group_id`` + ``track``). Every
   business frontmatter schema in ``infra/persistence/markdown/_mds/``
   subclasses one of these.

Concrete business schemas (``EpisodeDailyFrontmatter``,
``UserProfileFrontmatter``, …) live in ``infra``; they add per-record
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
    ``GroupScopedFrontmatter``) plus per-record business fields.
    """

    SCOPE_DIR: ClassVar[str] = ""
    """Top-level directory under the memory-root that holds this kind.

    Scope mixins set this to ``"users"`` / ``"groups"``. Scope-agnostic
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
        :class:`DailyLogPathMixin` or :class:`ProfilePathMixin` *before* the
        scope mixin in the MRO so this abstract version is shadowed.
        """
        raise NotImplementedError(
            f"{cls.__name__} must declare path_glob() "
            f"(mix in DailyLogPathMixin / ProfilePathMixin, or override directly)"
        )


class DailyLogPathMixin:
    """Path strategy for daily-log files.

    Files live at ``<SCOPE_DIR>/<scope_id>/<DIR_NAME>/<FILE_PREFIX>-<YYYY-MM-DD>.md``.
    Subclasses must inherit a scope mixin (``UserScopedFrontmatter`` /
    ``GroupScopedFrontmatter``) supplying ``SCOPE_DIR``, and must declare
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


class ProfilePathMixin:
    """Path strategy for single-file profile markdown.

    Profiles live at ``<SCOPE_DIR>/<scope_id>/<PROFILE_FILENAME>`` —
    one fixed-name file directly under the scope's owner directory, no
    intermediate ``<dir>/`` segment (unlike daily-logs). Subclasses must
    inherit a scope mixin (``UserScopedFrontmatter`` /
    ``GroupScopedFrontmatter``) supplying
    ``SCOPE_DIR`` and declare their own ``PROFILE_FILENAME``.

    Place **this mixin first** so MRO resolves ``path_glob()`` here::

        class UserProfileFrontmatter(ProfilePathMixin, UserScopedFrontmatter):
            PROFILE_FILENAME: ClassVar[str] = "user.md"
            ...

    A kind that keeps several files of one schema under the same scope
    directory declares ``PROFILE_GLOB`` too, because the cascade scanner
    globs each kind exactly once and ``*`` does not span ``/``. Both files
    must still sit directly under ``<scope_id>/``: a subdirectory would need
    an extra path component the single glob cannot express.
    """

    PROFILE_FILENAME: ClassVar[str]
    PROFILE_GLOB: ClassVar[str | None] = None
    """Filename glob covering every file of this kind; ``None`` means the
    kind has exactly one file and ``PROFILE_FILENAME`` is the pattern."""
    SCOPE_DIR: ClassVar[str]

    @classmethod
    def path_glob(cls) -> str:
        # Leading ``*/*/`` matches the <app>/<project> scope prefix.
        return f"*/*/{cls.SCOPE_DIR}/*/{cls.PROFILE_GLOB or cls.PROFILE_FILENAME}"


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


class GroupScopedFrontmatter(BaseFrontmatter):
    """Records that belong to a group (track = ``group``).

    The third owner type (docs/group-memory.md, D1): ``group_id`` is the
    file-level scope the path already expresses, as ``user_id`` is for
    :class:`UserScopedFrontmatter`. Who said what inside the group rides on
    each entry (``sender_ids`` / ``subject_ids`` / ``origin``), never here.
    """

    SCOPE_DIR: ClassVar[str] = "groups"

    group_id: str
    track: Literal["group"] = "group"
