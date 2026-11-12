"""Profile frontmatter -- one person's profile in one space, one schema.

Two files carry it (docs/group-memory.md, 7.4):

- ``users/<user_id>/user.md`` (:class:`UserProfileFrontmatter`): a user's
  profile in their own one-to-one memory;
- ``groups/<group_id>/members/<member_id>.md``
  (:class:`GroupMemberProfileFrontmatter`): a member's profile inside a
  group, read only from that group's segments.

Both are :class:`ProfileFrontmatter`: a free-form ``summary`` plus the two
evidence buckets :class:`everalgo.user_memory.ProfileExtractor` emits
(``explicit_info`` / ``implicit_traits``), every item carrying its
provenance; ``profile_timestamp_ms`` mirrors
:attr:`everalgo.types.Profile.timestamp`; ``history`` keeps the items updates
replaced or retired; ``pending_memcell_ids`` are segments no update has read
yet (a group's cadence). The two bindings add only the path, the owner keys
and what each file leaves out: ``user.md`` writes ``pending_memcell_ids`` and
``history`` only while they hold something, and a one-to-one profile has
nothing pending (it reads every segment at once). The body is the summary,
for a human reader.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from everos.infra.persistence.markdown._base.frontmatter import (
    BaseFrontmatter,
    GroupScopedFrontmatter,
    ProfilePathMixin,
    UserScopedFrontmatter,
)


def _empty(value: list[Any]) -> bool:
    return not value


class ProfileFrontmatter(BaseFrontmatter):
    """The fields of a person's profile, in either space."""

    summary: str = ""
    """Free-form one-paragraph summary of the person — the retrieval anchor."""

    explicit_info: list[Any] = Field(default_factory=list)
    """Algo-side ``explicit_info`` bucket (verbatim facts the person stated)."""

    implicit_traits: list[Any] = Field(default_factory=list)
    """Algo-side ``implicit_traits`` bucket (LLM-inferred preferences)."""

    profile_timestamp_ms: int = 0
    """Algo-emitted profile timestamp (ms epoch): the timestamp of the most
    recent segment an update read."""

    primary_language: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """The ``OutputLanguage`` value the profile is frozen to; left out of the
    file while unknown, and absent from profiles written before it."""

    pending_memcell_ids: list[str] = Field(default_factory=list)
    """Segments the person spoke in that no update has read yet."""

    history: list[Any] = Field(default_factory=list)
    """Items an update replaced or retired, newest first, each with
    ``replaced_at`` and what replaced it."""


class UserProfileFrontmatter(
    ProfilePathMixin, ProfileFrontmatter, UserScopedFrontmatter
):
    """``users/<user_id>/user.md``: the user's own profile."""

    PROFILE_FILENAME: ClassVar[str] = "user.md"

    type: Literal["user_profile"] = "user_profile"
    pending_memcell_ids: list[str] = Field(default_factory=list, exclude_if=_empty)
    history: list[Any] = Field(default_factory=list, exclude_if=_empty)


class _MemberScopedFrontmatter(GroupScopedFrontmatter):
    """A file of one member inside a group: the group's scope plus the member."""

    member_id: str


class GroupMemberProfileFrontmatter(ProfileFrontmatter, _MemberScopedFrontmatter):
    """``groups/<group_id>/members/<member_id>.md``: a member's profile in a group."""

    MEMBERS_DIR: ClassVar[str] = "members"

    type: Literal["group_member_profile"] = "group_member_profile"

    @classmethod
    def path_glob(cls) -> str:
        # Leading ``*/*/`` matches the <app>/<project> scope prefix.
        return f"*/*/{cls.SCOPE_DIR}/*/{cls.MEMBERS_DIR}/*.md"
