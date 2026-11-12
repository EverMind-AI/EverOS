"""UserProfile frontmatter — single-file profile markdown for users.

Path: ``users/<user_id>/user.md``.

Carries the LLM-synthesised user profile: a free-form ``summary`` plus the
two evidence buckets emitted by :class:`everalgo.user_memory.ProfileExtractor`
(``explicit_info`` / ``implicit_traits``). ``profile_timestamp_ms``
mirrors :attr:`everalgo.types.Profile.timestamp`.

``do_not_record`` and ``pending`` are the extractor's own state, kept here so
the next extraction reads them back. They are left out of the file while
empty, so a profile without them reads and writes as before, and they never
reach the derived index or the API.
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

    do_not_record: list[dict[str, Any]] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    """The extractor's do-not-record list, verbatim: the matters the user
    asked not to be recorded."""

    pending: list[dict[str, Any]] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    """The extractor's pending list, verbatim: pattern items waiting for a
    later conversation to repeat them."""
