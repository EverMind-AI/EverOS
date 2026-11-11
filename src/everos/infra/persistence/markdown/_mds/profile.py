"""UserProfile frontmatter — single-file profile markdown for users.

Path: ``users/<user_id>/user.md``.

Carries the LLM-synthesised user profile: a free-form ``summary`` plus the
two evidence buckets emitted by :class:`everalgo.user_memory.ProfileExtractor`
(``explicit_info`` / ``implicit_traits``). ``profile_timestamp_ms``
mirrors :attr:`everalgo.types.Profile.timestamp`, and ``history`` keeps the
items updates replaced or retired, as a member's profile in a group does
(docs/group-memory.md, 7.4).
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
    """Algo-emitted profile timestamp (ms epoch): the timestamp of the most
    recent segment an update read."""

    primary_language: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The ``OutputLanguage`` value the profile is frozen to; left out of the
    file while unknown, and absent from profiles written before it."""

    history: list[Any] = Field(default_factory=list, exclude_if=lambda value: not value)
    """Items an update replaced or retired, newest first, each with
    ``replaced_at`` and what replaced it; left out of the file while empty."""
