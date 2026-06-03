"""Tests for the profile frontmatter duck-typed shape.

Profile schemas have no shared base class — they only need a
``PROFILE_FILENAME`` ClassVar plus inheritance from a scope mixin. This
test exercises that contract via a local fixture class.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pytest
from pydantic import ValidationError

from everos.core.persistence.markdown import UserScopedFrontmatter


class _SampleUserProfileFM(UserScopedFrontmatter):
    """Local fixture: a user-track profile schema."""

    PROFILE_FILENAME: ClassVar[str] = "user.md"

    type: Literal["sample_user_profile"] = "sample_user_profile"
    display_name: str
    bio: str
    interests: list[str] = []


def test_schema_inherits_user_scope() -> None:
    fm = _SampleUserProfileFM(
        id="sample_user_profile_u_jason",
        type="sample_user_profile",
        user_id="u_jason",
        display_name="Jason",
        bio="hiker.",
    )
    assert fm.track == "user"
    assert fm.SCOPE_DIR == "users"


def test_profile_filename_classvar() -> None:
    """Path-shape ClassVar is duck-typed onto the schema directly."""
    assert _SampleUserProfileFM.PROFILE_FILENAME == "user.md"


def test_requires_display_name_and_bio() -> None:
    with pytest.raises(ValidationError):
        _SampleUserProfileFM(  # type: ignore[call-arg]
            id="x",
            type="sample_user_profile",
            user_id="u_jason",
            bio="missing display_name",
        )
    with pytest.raises(ValidationError):
        _SampleUserProfileFM(  # type: ignore[call-arg]
            id="x",
            type="sample_user_profile",
            user_id="u_jason",
            display_name="missing bio",
        )


def test_interests_default_empty() -> None:
    fm = _SampleUserProfileFM(
        id="x",
        type="sample_user_profile",
        user_id="u_jason",
        display_name="Jason",
        bio="hiker.",
    )
    assert fm.interests == []
