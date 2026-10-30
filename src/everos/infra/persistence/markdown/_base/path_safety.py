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
Some callers (``knowledge_writer.py``) concatenate the result directly under a
shared directory with no per-caller prefix, so the guarantee below has to hold
on its own, without relying on a prefix like ``skill_`` to absorb a degenerate
result.

``sanitize_dirname`` is idempotent (``sanitize_dirname(sanitize_dirname(x),
fb) == sanitize_dirname(x, fb)``): a name built by re-sanitizing an
already-sanitized segment (e.g. one derived by walking the directory tree)
lands on the same string as sanitizing the original raw name. That property
is what lets a reader and a writer agree on a path even when one side has
only the raw name and the other only the on-disk directory name.
"""

from __future__ import annotations

import re
import unicodedata

_MAX_DIRNAME_LEN = 50
_SAFE_CHARS = re.compile(r"[^\w\-.]", re.UNICODE)
_DEGENERATE = frozenset({"", ".", ".."})


def sanitize_dirname(raw: str, fallback: str) -> str:
    """Produce a safe directory/file name segment from free-text input.

    * NFC-normalize first. For an ordinary decomposed (NFD) input — a base
      letter plus a combining mark, e.g. ``"e"`` + combining acute accent —
      this collapses to the precomposed form before the character filter
      runs, so the accent survives (a combining mark alone is not ``\\w``
      and would otherwise be silently stripped). This is best-effort, not a
      guarantee: for the ~1,082 Unicode *composition exclusion* codepoints
      (e.g. Devanagari ``क़``/``ख़``, U+0958/U+0959), NFC does the
      opposite — it *decomposes* an already-precomposed exclusion
      character, because recomposing it is explicitly excluded from the
      NFC algorithm, and the resulting combining mark is then stripped just
      the same. Normalizing here improves fidelity for the common case; it
      does not make every Unicode script round-trip losslessly.
    * Replace spaces with underscores.
    * Strip characters outside ``[a-zA-Z0-9_\\-.]`` (``\\w`` is Unicode-aware,
      so CJK and other non-ASCII scripts survive readably). Note that ``.``
      is a *safe* character, not stripped — a run of literal dots is a legal
      result of this step.
    * Truncate to 50 characters.
    * Fall back to *fallback* if the result is empty, ``"."``, or ``".."``.

    Every path separator (``/``, ``\\``) is stripped by the character-class
    filter, so no separator survives and the result is always exactly one
    path component — it can never be split into multiple segments by a
    downstream ``Path(...) / result``. The fallback on ``""`` / ``"."`` /
    ``".."`` closes the remaining gap: those are the only single components
    that resolve to *no new child* (``""`` and ``"."`` both mean "this same
    directory", ``".."`` means "its parent") rather than a genuinely new
    entry. With both guarantees together, ``Path(some_dir) / sanitize_dirname(raw, fb)``
    can never escape ``some_dir`` and never silently collapses back onto it
    or its parent — unconditionally, including for a caller with no
    additional prefix (like ``skill_``) protecting the segment.

    This function is lossy and not injective: distinct inputs can sanitize
    to the same output (dropped characters, space/underscore collapse, and
    truncation are all many-to-one). A caller that needs distinct outputs
    for distinct inputs must disambiguate before or after calling this —
    the function itself makes no such guarantee.
    """
    slug = unicodedata.normalize("NFC", raw)
    slug = slug.replace(" ", "_")
    slug = _SAFE_CHARS.sub("", slug)
    slug = slug[:_MAX_DIRNAME_LEN]
    return slug if slug not in _DEGENERATE else fallback
