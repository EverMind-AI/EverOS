"""UserProfile frontmatter — single-file profile markdown for users.

Path: ``users/<user_id>/user.md``.

Carries the LLM-synthesised user profile: a free-form ``summary`` plus the
two evidence buckets emitted by :class:`everalgo.user_memory.ProfileExtractor`
(``explicit_info`` / ``implicit_traits``). ``profile_timestamp_ms``
mirrors :attr:`everalgo.types.Profile.timestamp`.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from everos.infra.persistence.markdown._base.frontmatter import (
    ProfilePathMixin,
    UserScopedFrontmatter,
)


class UserProfileFrontmatter(ProfilePathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<user_id>/user.md``."""

    PROFILE_FILENAME: ClassVar[str] = "user.md"

    type: Literal["user_profile"] = "user_profile"

    summary: str = ""
    """Free-form one-paragraph summary of the user — the retrieval anchor."""

    explicit_info: list[Any] = []
    """Algo-side ``explicit_info`` bucket (verbatim facts the user stated)."""

    implicit_traits: list[Any] = []
    """Algo-side ``implicit_traits`` bucket (LLM-inferred preferences)."""

    profile_timestamp_ms: int = 0
    """Algo-emitted profile timestamp (ms epoch): the timestamp of the
    newest episode that fed into the synthesis."""

    primary_language: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The ``OutputLanguage`` value the profile is frozen to; left out of the
    file while unknown, and absent from profiles written before it."""
