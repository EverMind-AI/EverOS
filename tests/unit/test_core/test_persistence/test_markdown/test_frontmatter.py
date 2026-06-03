"""Unit tests for frontmatter parse / dump + path_glob chassis."""

from __future__ import annotations

from typing import ClassVar, Literal

import pytest

from everos.core.persistence import (
    AgentScopedFrontmatter,
    BaseFrontmatter,
    DailyLogPathMixin,
    SkillPathMixin,
    UserScopedFrontmatter,
    dump_frontmatter,
    parse_frontmatter,
)


def test_parse_no_frontmatter() -> None:
    text = "# Just a heading\n\nbody."
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_empty_frontmatter() -> None:
    text = "---\n---\n# body\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == "# body\n"


def test_parse_simple_frontmatter() -> None:
    text = "---\ntitle: Hello\ntags: [a, b]\n---\n# body\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"title": "Hello", "tags": ["a", "b"]}
    assert body == "# body\n"


def test_parse_nested_frontmatter() -> None:
    text = "---\nuser:\n  id: u_1\n  name: Alice\n---\nbody"
    meta, body = parse_frontmatter(text)
    assert meta == {"user": {"id": "u_1", "name": "Alice"}}
    assert body == "body"


def test_parse_no_closing_delim() -> None:
    """Missing closing --- → treat as no frontmatter (return original text)."""
    text = "---\ntitle: Hello\n# body without closing\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_non_mapping_yaml() -> None:
    """YAML that parses to a non-mapping (e.g. list) → empty dict + original text."""
    text = "---\n- item1\n- item2\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_opening_delim_no_newline() -> None:
    """``---`` followed by non-newline char → not a frontmatter block."""
    text = "---this is not frontmatter"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_unicode_values() -> None:
    text = "---\ntitle: 你好\n---\n世界"
    meta, body = parse_frontmatter(text)
    assert meta == {"title": "你好"}
    assert body == "世界"


def test_dump_empty_mapping_returns_empty_string() -> None:
    assert dump_frontmatter({}) == ""


def test_dump_simple_mapping() -> None:
    out = dump_frontmatter({"title": "Hello"})
    assert out.startswith("---\n")
    assert out.endswith("---\n")
    assert "title: Hello" in out


def test_dump_preserves_key_order() -> None:
    out = dump_frontmatter({"z": 1, "a": 2, "m": 3})
    body = out.strip("-\n")
    keys = [line.split(":", 1)[0] for line in body.strip().splitlines() if ":" in line]
    assert keys == ["z", "a", "m"]


def test_dump_unicode() -> None:
    out = dump_frontmatter({"title": "你好"})
    assert "你好" in out  # allow_unicode keeps non-ASCII verbatim


def test_round_trip() -> None:
    meta = {"title": "Hello", "tags": ["a", "b"], "nested": {"k": "v"}}
    body_text = "# Body\n\nLine.\n"
    composed = dump_frontmatter(meta) + body_text
    parsed_meta, parsed_body = parse_frontmatter(composed)
    assert parsed_meta == meta
    assert parsed_body == body_text


# ── path_glob chassis ───────────────────────────────────────────────────


def test_base_path_glob_raises_not_implemented() -> None:
    """A schema with no strategy mixin must surface a clear error."""

    class _PlainFm(BaseFrontmatter):
        type: Literal["_plain"] = "_plain"

    with pytest.raises(NotImplementedError, match="path_glob"):
        _PlainFm.path_glob()


def test_daily_log_path_glob_user_scope() -> None:
    """Mixin builds ``users/*/<dir>/<prefix>-*.md`` from ClassVars."""

    class _UserDaily(DailyLogPathMixin, UserScopedFrontmatter):
        DIR_NAME: ClassVar[str] = "demo"
        FILE_PREFIX: ClassVar[str] = "entry"
        type: Literal["_user_daily"] = "_user_daily"

    assert _UserDaily.path_glob() == "*/*/users/*/demo/entry-*.md"


def test_daily_log_path_glob_agent_scope() -> None:
    """Same mixin, agent scope swaps the leading directory."""

    class _AgentDaily(DailyLogPathMixin, AgentScopedFrontmatter):
        DIR_NAME: ClassVar[str] = "cases"
        FILE_PREFIX: ClassVar[str] = "case"
        type: Literal["_agent_daily"] = "_agent_daily"

    assert _AgentDaily.path_glob() == "*/*/agents/*/cases/case-*.md"


def test_skill_path_glob() -> None:
    """SkillPathMixin builds ``<scope>/*/<container>/<prefix>*/<main>``."""

    class _AgentSkill(SkillPathMixin, AgentScopedFrontmatter):
        SKILLS_CONTAINER_NAME: ClassVar[str] = "skills"
        SKILL_DIR_PREFIX: ClassVar[str] = "skill_"
        SKILL_MAIN_FILENAME: ClassVar[str] = "SKILL.md"
        type: Literal["_agent_skill"] = "_agent_skill"

    assert _AgentSkill.path_glob() == "*/*/agents/*/skills/skill_*/SKILL.md"


def test_strategy_mixin_overrides_base_via_mro() -> None:
    """Strategy mixin placed first in the parent list wins over abstract base."""

    class _Daily(DailyLogPathMixin, UserScopedFrontmatter):
        DIR_NAME: ClassVar[str] = "x"
        FILE_PREFIX: ClassVar[str] = "y"
        type: Literal["_daily_mro"] = "_daily_mro"

    # Concrete is reachable; abstract NotImplementedError is shadowed.
    assert isinstance(_Daily.path_glob(), str)
    assert "NotImplementedError" not in _Daily.path_glob()
