"""Unit tests for :func:`sanitize_dirname` — the shared CWE-22 path-safety helper.

Pins three properties the callers rely on:

- traversal payloads collapse to a segment with no path separator and no
  ``..`` component (the actual security property);
- non-ASCII input (CJK, spaces) survives readably rather than being
  sanitized down to the empty-string fallback;
- the function is idempotent, which is what lets a reader (deriving a name
  from an on-disk directory) and a writer (deriving it from raw input)
  agree on the same path — see :mod:`.test_frontmatter`'s
  ``SkillPathMixin.skill_dir_name`` coverage for the consumer side.
"""

from __future__ import annotations

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
