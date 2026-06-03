"""Tests for :class:`ProfileReader` — typed read for profile files."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import ValidationError

from everos.core.persistence import MemoryRoot, UserScopedFrontmatter
from everos.infra.persistence.markdown.readers import ProfileReader
from everos.infra.persistence.markdown.writers import ProfileWriter


class _UserProfileFM(UserScopedFrontmatter):
    PROFILE_FILENAME: ClassVar[str] = "user.md"
    type: Literal["demo_user_profile"] = "demo_user_profile"
    display_name: str = ""
    bio: str = ""
    interests: list[str] = []


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


@pytest.fixture
def writer(root: MemoryRoot) -> ProfileWriter:
    return ProfileWriter(root)


@pytest.fixture
def reader(root: MemoryRoot) -> ProfileReader:
    return ProfileReader(root)


async def test_read_returns_typed_frontmatter_and_body(
    writer: ProfileWriter, reader: ProfileReader
) -> None:
    fm_in = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
        display_name="Jason",
        bio="weekend hiker.",
        interests=["hiking", "coffee"],
    )
    await writer.write("u_jason", frontmatter=fm_in, body="The body.")

    out = await reader.read("u_jason", schema=_UserProfileFM)
    assert out is not None
    fm_out, body = out
    assert isinstance(fm_out, _UserProfileFM)
    assert fm_out.display_name == "Jason"
    assert fm_out.interests == ["hiking", "coffee"]
    assert body == "The body."


async def test_read_returns_none_when_missing(reader: ProfileReader) -> None:
    assert await reader.read("u_ghost", schema=_UserProfileFM) is None


async def test_read_round_trip_through_extra_fields(
    writer: ProfileWriter, reader: ProfileReader
) -> None:
    """L2 / L4 ride-along fields survive a write+read cycle."""
    fm_in = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
        md_sha256="abc",  # extra
        custom_label="ride-along",  # extra
    )
    await writer.write("u_jason", frontmatter=fm_in, body="b")
    out = await reader.read("u_jason", schema=_UserProfileFM)
    assert out is not None
    fm_out, _ = out
    dumped = fm_out.model_dump()
    assert dumped["md_sha256"] == "abc"
    assert dumped["custom_label"] == "ride-along"


async def test_read_validates_against_supplied_schema(
    writer: ProfileWriter, reader: ProfileReader
) -> None:
    """A stricter schema rejects loose existing data — proves typed parsing."""

    class _StricterFM(UserScopedFrontmatter):
        PROFILE_FILENAME: ClassVar[str] = "user.md"
        type: Literal["demo_user_profile"] = "demo_user_profile"
        # Required field with no default — written file lacks it.
        priority: int

    fm_in = _UserProfileFM(
        id="demo_user_profile_u_jason",
        type="demo_user_profile",
        user_id="u_jason",
    )
    await writer.write("u_jason", frontmatter=fm_in, body="b")

    with pytest.raises(ValidationError):
        await reader.read("u_jason", schema=_StricterFM)


def test_path_for_matches_writer(
    tmp_path: Path,
    writer: ProfileWriter,
    reader: ProfileReader,
) -> None:
    """Reader and writer resolve to the same path for the same schema."""
    assert reader.path_for("u_jason", schema=_UserProfileFM) == writer.path_for(
        "u_jason", schema=_UserProfileFM
    )


def test_path_for_does_not_create_files(tmp_path: Path, reader: ProfileReader) -> None:
    p = reader.path_for("u_jason", schema=_UserProfileFM)
    assert not p.exists()
    assert not (tmp_path / "users").exists()
