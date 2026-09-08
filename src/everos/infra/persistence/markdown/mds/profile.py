"""UserProfile frontmatter — single-file profile markdown for users.

Path: ``users/<user_id>/user.md``.

Carries the LLM-synthesised user profile: a free-form ``summary`` plus the
two evidence buckets emitted by :class:`everalgo.user_memory.ProfileExtractor`
(``explicit_info`` / ``implicit_traits``). ``profile_timestamp_ms``
mirrors :attr:`everalgo.types.Profile.timestamp` so the
``extract_user_profile`` strategy can compare per-user freshness against
cluster ``last_ts`` without re-parsing the body.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from everos.core.persistence.markdown import ProfilePathMixin, UserScopedFrontmatter


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
    """Algo-emitted profile timestamp (ms epoch); equals the timestamp of
    the most recent MemCell that fed into the synthesis. Compared with
    :attr:`everos.infra.persistence.sqlite.Cluster.last_ts_ms` to decide
    whether a cluster is fresh enough to drive a profile re-extraction."""
