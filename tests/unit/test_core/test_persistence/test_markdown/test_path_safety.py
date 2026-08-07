"""Unit tests for :func:`sanitize_dirname` — the shared CWE-22 path-safety helper.

Pins the properties the callers rely on:

- traversal payloads collapse to a single opaque path component with no
  separator (the "no separator survives" half of the safety property);
- short inputs that are themselves sanitizer fixpoints — bare ``".."``, or
  anything that strips down to ``".."`` / ``"."`` once a separator is
  removed — fall back rather than being returned as-is (the other half:
  without this, ``sanitize_dirname("../", fb)`` returns ``".."`` verbatim,
  which is a real one-level escape for a caller with no additional prefix
  protecting the segment, like ``KnowledgeWriter``);
- non-ASCII input (CJK, spaces, NFD-decomposed accents) survives readably
  rather than being sanitized down to the empty-string fallback;
- the function is idempotent, which is what lets a reader (deriving a name
  from an on-disk directory) and a writer (deriving it from raw input)
  agree on the same path — see :mod:`.test_frontmatter`'s
  ``SkillPathMixin.skill_dir_name`` coverage for the consumer side.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from everos.core.persistence.markdown import sanitize_dirname


def test_traversal_payload_has_no_separator() -> None:
    """No path separator survives — a run of literal dots (``....``) is a
    single opaque filename component, not a ``..`` path-traversal segment,
    since there is no ``/`` left to divide it into components.
    """
    payload = "../" * 8 + "tmp/pwned"
    sanitized = sanitize_dirname(payload, fallback="unnamed")

    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert sanitized != ".."


def test_traversal_payload_resolved_path_stays_under_root(tmp_path: Path) -> None:
    payload = "../" * 8 + "tmp/pwned"
    sanitized = sanitize_dirname(payload, fallback="unnamed")

    resolved = (tmp_path / sanitized).resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("raw", ["..", "../", "/../", ".", "./"])
def test_degenerate_fixpoints_fall_back_instead_of_escaping(raw: str) -> None:
    """A short input that strips down to exactly ``".."`` or ``"."`` must
    fall back, not be returned as-is.

    Regression guard for the actual bug: ``"."`` is a *safe* character
    (kept, not stripped), so ``"../"`` and ``"."``+``"/"`` both collapse to
    ``".."`` / ``"."`` once the separator is removed — a fixpoint of the
    old (empty-only) fallback check, since neither is the empty string.
    Without this fallback, a caller with no extra prefix protecting the
    segment (``KnowledgeWriter``, unlike the skill writer's ``skill_``
    prefix) resolves one directory level up or sideways instead of into a
    new child.
    """
    sanitized = sanitize_dirname(raw, fallback="unnamed")
    assert sanitized == "unnamed"


def test_knowledge_style_unprefixed_concatenation_stays_under_root() -> None:
    """The one-level escape the coordinator reproduced on the (unprefixed)
    knowledge path: ``Path(root) / sanitize_dirname("../", fb) / "doc_123"``
    must resolve under ``root``, not to ``root``'s parent.
    """
    root = Path("/root/knowledge")
    resolved = root / sanitize_dirname("../", "Others") / "doc_123"
    assert resolved == Path("/root/knowledge/Others/doc_123")


def test_nfc_normalizes_decomposed_accents() -> None:
    """An NFD-decomposed accented character (base letter + combining mark)
    must sanitize to the same result as its NFC (precomposed) form —
    without normalization, the combining mark is not ``\\w`` and gets
    silently stripped, losing the accent instead of preserving it.
    """
    nfc = "café"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd  # sanity: the two forms really are distinct strings

    sanitized_nfc = sanitize_dirname(nfc, fallback="unnamed")
    sanitized_nfd = sanitize_dirname(nfd, fallback="unnamed")

    assert sanitized_nfc == sanitized_nfd == "café"


def test_nfc_does_not_help_composition_exclusions() -> None:
    """Pins the documented exception: for Unicode "composition exclusion"
    codepoints, NFC normalization does not help — it decomposes an
    already-precomposed character, and the resulting combining mark is
    stripped either way.

    U+0958 / U+0959 (Devanagari letters formed from a base letter + nukta)
    are composition exclusions: their canonical decomposition is excluded
    from NFC recomposition, so ``normalize("NFC", precomposed)`` yields the
    *decomposed* form, not the precomposed one.
    """
    precomposed = "क़ख़"
    assert unicodedata.normalize("NFC", precomposed) != precomposed

    sanitized = sanitize_dirname(precomposed, fallback="unnamed")

    # The nukta (combining mark, U+093C) is lost: NFC decomposes the
    # precomposed input into base + nukta, and the nukta is then stripped
    # (not \w) -- the opposite of what NFC does for an ordinary NFD accent.
    assert sanitized == "कख"


def test_cjk_and_space_input_preserved_readably() -> None:
    raw = "修复 Django 自动重载问题"
    sanitized = sanitize_dirname(raw, fallback="unnamed")

    assert "修复" in sanitized
    assert "Django" in sanitized
    assert "_" in sanitized  # spaces became underscores, not stripped
    assert " " not in sanitized


@pytest.mark.parametrize(
    "raw",
    [
        "../" * 8 + "tmp/pwned",
        "修复 Django 自动重载问题",
        "normal_skill",
        "../../etc/passwd",
        "   ",
        "!!!@@@###",
        "..",
        "../",
        "/../",
        ".",
        "./",
    ],
)
def test_sanitize_is_idempotent(raw: str) -> None:
    once = sanitize_dirname(raw, fallback="unnamed")
    twice = sanitize_dirname(once, fallback="unnamed")
    assert once == twice


def test_empty_result_falls_back() -> None:
    assert sanitize_dirname("!!!@@@###", fallback="unnamed") == "unnamed"


def test_truncates_to_max_length() -> None:
    sanitized = sanitize_dirname("a" * 200, fallback="unnamed")
    assert len(sanitized) == 50
