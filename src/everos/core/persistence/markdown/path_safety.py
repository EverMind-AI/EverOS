"""``sanitize_dirname`` — the single path-safety primitive for md directory names.

Several markdown layouts turn free-text into a filesystem path segment:
knowledge document/category titles, and agent-skill names. Both sources are
untrusted in the same way — knowledge titles come from parsed source
documents, skill names come straight from LLM output — so a name containing
``../`` or a path separator must never survive into a directory segment
(CWE-22 path traversal).

This module is the one place that decision is made. Callers that need a
filesystem-safe segment from a free-text string route through
:func:`sanitize_dirname` rather than keeping a private regex copy — see
``writers/knowledge_writer.py`` and
:meth:`SkillPathMixin.skill_dir_name() <.frontmatter.SkillPathMixin.skill_dir_name>`.

``sanitize_dirname`` is idempotent (``sanitize_dirname(sanitize_dirname(x),
fb) == sanitize_dirname(x, fb)``): a name built by re-sanitizing an
already-sanitized segment (e.g. one derived by walking the directory tree)
lands on the same string as sanitizing the original raw name. That property
is what lets a reader and a writer agree on a path even when one side has
only the raw name and the other only the on-disk directory name.
"""

from __future__ import annotations

import re

_MAX_DIRNAME_LEN = 50
_SAFE_CHARS = re.compile(r"[^\w\-.]", re.UNICODE)


def sanitize_dirname(raw: str, fallback: str) -> str:
    """Produce a safe directory/file name segment from free-text input.

    * Replace spaces with underscores.
    * Strip characters outside ``[a-zA-Z0-9_\\-.]`` (``\\w`` is Unicode-aware,
      so CJK and other non-ASCII scripts survive readably).
    * Truncate to 50 characters.
    * Fall back to *fallback* if the result is empty.

    Path separators (``/``, ``\\``) and ``..`` sequences are always stripped
    by the character class above, so a traversal payload like
    ``"../../../../tmp/pwned"`` collapses to a harmless run of dots and
    letters with no separator — it cannot escape the directory it is
    concatenated into.
    """
    slug = raw.replace(" ", "_")
    slug = _SAFE_CHARS.sub("", slug)
    slug = slug[:_MAX_DIRNAME_LEN]
    return slug if slug else fallback
