"""Tests for :class:`ProfileWriter` — single-file rewrite layout."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import pytest

from everos.core.persistence import (
    AgentScopedFrontmatter,
    BaseFrontmatter,
    MarkdownReader,
    MemoryRoot,
    UserScopedFrontmatter,
)
from everos.infra.persistence.markdown.writers import ProfileWriter


class _UserProfileFM(UserScopedFrontmatter):
    PROFILE_FILENAME: ClassVar[str] = "user.md"
    type: Literal["demo_user_profile"] = "demo_user_profile"
    display_name: str = ""
    bio: str = ""


class _AgentProfileFM(AgentScopedFrontmatter):
    PROFILE_FILENAME: ClassVar[str] = "agent.md"
    type: Literal["demo_agent_profile"] = "demo_agent_profile"
    name: str = ""


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


@pytest.fixture
def writer(root: MemoryRoot) -> ProfileWriter:
    return ProfileWriter(root)


async def test_write_creates_user_profile(
    root: MemoryRoot, writer: ProfileWriter
) -> None:
    fm = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
        display_name="Jason",
        bio="hiker.",
    )
    path = await writer.write("u_jason", frontmatter=fm, body="Long-form profile.")
    expected = root.users_dir() / "u_jason" / "user.md"
    assert path == expected
    assert expected.is_file()


async def test_write_creates_agent_profile(
    root: MemoryRoot, writer: ProfileWriter
) -> None:
    fm = _AgentProfileFM(
        id="demo_agent_profile_agent_x",
        type="demo_agent_profile",
        agent_id="agent_x",
        name="zhang_legal",
    )
    path = await writer.write("agent_x", frontmatter=fm, body="Agent playbook.")
    expected = root.agents_dir() / "agent_x" / "agent.md"
    assert path == expected
    assert expected.is_file()


async def test_write_writes_frontmatter_and_body(
    root: MemoryRoot, writer: ProfileWriter
) -> None:
    fm = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
        display_name="Jason",
        bio="weekend hiker.",
    )
    await writer.write("u_jason", frontmatter=fm, body="The body.")

    parsed = await MarkdownReader.read(root.users_dir() / "u_jason" / "user.md")
    assert parsed.frontmatter["display_name"] == "Jason"
    assert parsed.frontmatter["bio"] == "weekend hiker."
    assert parsed.body.rstrip("\n") == "The body."


async def test_write_is_upsert_full_replace(
    root: MemoryRoot, writer: ProfileWriter
) -> None:
    """Second call overwrites both frontmatter and body — no append."""
    fm1 = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
        display_name="Jason v1",
        bio="v1",
    )
    await writer.write("u_jason", frontmatter=fm1, body="body v1")

    fm2 = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
        display_name="Jason v2",
        bio="v2",
    )
    await writer.write("u_jason", frontmatter=fm2, body="body v2")

    parsed = await MarkdownReader.read(root.users_dir() / "u_jason" / "user.md")
    assert parsed.frontmatter["display_name"] == "Jason v2"
    assert parsed.frontmatter["bio"] == "v2"
    assert parsed.body.rstrip("\n") == "body v2"
    assert "v1" not in parsed.body


def test_path_for_does_not_create_files(
    root: MemoryRoot, writer: ProfileWriter
) -> None:
    """``path_for`` is a pure path resolver — no IO."""
    p = writer.path_for("u_jason", schema=_UserProfileFM)
    assert p == root.users_dir() / "u_jason" / "user.md"
    assert not p.exists()
    assert not root.users_dir().exists()


async def test_write_normalises_trailing_newline(
    root: MemoryRoot, writer: ProfileWriter
) -> None:
    fm = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
    )
    await writer.write("u_jason", frontmatter=fm, body="no-newline-end")
    text = (root.users_dir() / "u_jason" / "user.md").read_text(encoding="utf-8")
    assert text.endswith("no-newline-end\n")


async def test_write_rejects_schema_missing_profile_filename(
    writer: ProfileWriter,
) -> None:
    """Schema without ``PROFILE_FILENAME`` ClassVar raises a clear error."""

    class _BadSchema(UserScopedFrontmatter):
        type: Literal["bad"] = "bad"

    fm = _BadSchema(id="x", type="bad", user_id="u_jason")
    with pytest.raises(TypeError, match="PROFILE_FILENAME"):
        await writer.write("u_jason", frontmatter=fm, body="body")


async def test_write_rejects_schema_missing_scope_dir(writer: ProfileWriter) -> None:
    """Schema without scope mixin (empty ``SCOPE_DIR``) raises a clear error."""

    class _ScopelessSchema(BaseFrontmatter):
        PROFILE_FILENAME: ClassVar[str] = "profile.md"
        type: Literal["scopeless"] = "scopeless"

    fm = _ScopelessSchema(id="x", type="scopeless")
    with pytest.raises(TypeError, match="SCOPE_DIR"):
        await writer.write("x", frontmatter=fm, body="body")
