"""Profile recall — a person's profile in a space, fetched by key, no ranking.

A profile is one row per person and space (no per-day fan-out, no entry
markers), so a read is a key lookup, never a ranking: a user's own profile by
their id, a member's profile in a group by the group and the member. One fetch
serves both read paths -- ``/search`` (``include_profile``, through
:class:`ProfileRecaller`) and ``/memory/get`` (``memory_type = profile`` /
``member_profile``) -- and gives both the same item fields
(:meth:`ProfileView.fields`), ``profile_data`` included. There is no
``query`` and no ``score``: the search DTO's optional ``score`` is reserved
for a future query-aware lookup.

A member's profile is cut to the asker's windows
(:func:`~everos.memory.access.visible_profile`): items whose sources lie
outside them are dropped, and with them the summary. The cascade keeps the
index in sync with the markdown (``users/<u>/user.md``,
``groups/<g>/members/<m>.md``).
"""

from __future__ import annotations

__all__ = [
    "ProfileRecaller",
    "ProfileView",
    "fetch_member_profile",
    "fetch_user_profile",
    "profile_data",
]

from dataclasses import dataclass

from everos.core.observability.logging import get_logger
from everos.memory.access import ReadScope, visible_profile
from everos.memory.models import Profile
from everos.memory.ports import ProfileIndex, Scope

from ..dto import SearchProfileItem

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProfileView:
    """A profile as a read shows it.

    Attributes:
        id: The index row (the user's id, or ``<group>:<member>``).
        app_id: The row's app.
        project_id: The row's project.
        profile: What the asker may see of the profile.
        group_id: The group a member's profile belongs to; ``None`` for a
            user's own.
        member_id: The member it is about; ``None`` for a user's own.
    """

    id: str
    app_id: str
    project_id: str
    profile: Profile
    group_id: str | None = None
    member_id: str | None = None

    def fields(self) -> dict[str, object]:
        """The fields a search item and a get item both carry."""
        if self.group_id is None:
            return {
                "id": self.id,
                "user_id": self.profile.owner_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
                "profile_data": profile_data(self.profile),
            }
        return {
            "id": self.id,
            "user_id": None,
            "owner_type": "group",
            "group_id": self.group_id,
            "member_id": self.member_id,
            "app_id": self.app_id,
            "project_id": self.project_id,
            "profile_data": profile_data(self.profile),
        }


async def fetch_user_profile(index: ProfileIndex, scope: Scope) -> ProfileView | None:
    """The user's own profile (``scope`` is theirs), or ``None`` before one exists."""
    stored = await index.get(scope)
    if stored is None:
        return None
    return ProfileView(
        id=stored.ref.id,
        app_id=stored.scope.app_id,
        project_id=stored.scope.project_id,
        profile=stored.item,
    )


async def fetch_member_profile(
    index: ProfileIndex, read: ReadScope, group_id: str, member_id: str
) -> ProfileView | None:
    """A member's profile in one group of ``read``, as the asker may see it.

    ``None`` when the member has no profile in the group, or when nothing of
    it lies in the asker's windows.
    """
    stored = await index.member(read.scope_of(group_id), member_id)
    if stored is None:
        return None
    shown = visible_profile(read, group_id, stored.item)
    if shown is None:
        return None
    return ProfileView(
        id=stored.ref.id,
        app_id=stored.scope.app_id,
        project_id=stored.scope.project_id,
        profile=shown,
        group_id=group_id,
        member_id=member_id,
    )


def profile_data(profile: Profile) -> dict[str, object]:
    """The ``profile_data`` mapping a profile item carries."""
    return {
        "summary": profile.summary,
        "explicit_info": [i.model_dump() for i in profile.explicit_info],
        "implicit_traits": [i.model_dump() for i in profile.implicit_traits],
        "profile_timestamp_ms": profile.timestamp,
    }


class ProfileRecaller:
    """Profiles for a search's ``include_profile``, as search items.

    Args:
        index: The profile index port.
    """

    def __init__(self, index: ProfileIndex) -> None:
        self._index = index

    async def fetch(
        self, owner_id: str, *, app_id: str, project_id: str
    ) -> list[SearchProfileItem]:
        """Return ``[item]`` if the user has a profile, otherwise ``[]``.

        Empty list (rather than 404) lets the caller emit a normal
        response with ``profiles=[]`` while the user is still in their
        cold-start window (no profile synthesised yet).
        """
        if not owner_id:
            return []
        found = await fetch_user_profile(
            self._index,
            Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=owner_id,
                owner_type="user",
            ),
        )
        if found is None:
            logger.debug("profile_fetch_miss", owner_id=owner_id)
            return []
        return [SearchProfileItem(**found.fields(), score=None)]

    async def fetch_member(
        self, read: ReadScope, group_id: str, member_id: str
    ) -> SearchProfileItem | None:
        """A member's profile in one group of ``read``, as the asker may see it."""
        found = await fetch_member_profile(self._index, read, group_id, member_id)
        if found is None:
            return None
        return SearchProfileItem(**found.fields(), score=None)
