"""Profile recall — KV fetch by (app_id, project_id, owner_id), no ranking.

Profile is the only owner-scoped kind that ships as **one row per
user** (no per-day fan-out, no entry markers). The recaller is a
deliberate KV-by-owner lookup: given ``owner_id``, return at most one
:class:`SearchProfileItem`. There is no ``query`` and no ``score``
field on the response — the DTO's optional ``score`` is reserved for
a future query-aware lookup.

The cascade keeps the profile index in sync with
``users/<user_id>/user.md``; this recaller reads the stored profile and
lays it out as the DTO's ``profile_data`` mapping (mirrors enterprise's
profile DTO shape).
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger
from everos.memory.access import ReadScope, visible_profile
from everos.memory.models import Profile
from everos.memory.ports import ProfileIndex, Scope

from ..dto import SearchProfileItem

logger = get_logger(__name__)


class ProfileRecaller:
    """Fetch the owner's profile from the profile index, return at most one item.

    Args:
        index: The profile index port.
    """

    def __init__(self, index: ProfileIndex) -> None:
        self._index = index

    async def fetch(
        self, owner_id: str, *, app_id: str, project_id: str
    ) -> list[SearchProfileItem]:
        """Return ``[item]`` if a profile row exists, otherwise ``[]``.

        Empty list (rather than 404) lets the caller emit a normal
        response with ``profiles=[]`` while the user is still in their
        cold-start window (no profile synthesised yet).
        """
        if not owner_id:
            return []
        stored = await self._index.get(
            Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=owner_id,
                owner_type="user",
            )
        )
        if stored is None:
            logger.debug("profile_fetch_miss", owner_id=owner_id)
            return []
        profile = stored.item
        return [
            SearchProfileItem(
                id=stored.ref.id,
                user_id=profile.owner_id,
                app_id=stored.scope.app_id,
                project_id=stored.scope.project_id,
                profile_data=profile_data(profile),
                score=None,
            )
        ]

    async def fetch_member(
        self, read: ReadScope, group_id: str, member_id: str
    ) -> SearchProfileItem | None:
        """A member's profile in one group of ``read``, as the asker may see it.

        Items whose sources lie outside the asker's windows are dropped (and
        with them the summary); ``None`` when nothing is left, or when the
        member has no profile in the group.
        """
        stored = await self._index.member(read.scope_of(group_id), member_id)
        if stored is None:
            return None
        shown = visible_profile(read, group_id, stored.item)
        if shown is None:
            return None
        return SearchProfileItem(
            id=stored.ref.id,
            user_id=None,
            owner_type="group",
            group_id=group_id,
            member_id=member_id,
            app_id=stored.scope.app_id,
            project_id=stored.scope.project_id,
            profile_data=profile_data(shown),
            score=None,
        )


def profile_data(profile: Profile) -> dict[str, object]:
    """The ``profile_data`` mapping a profile item carries."""
    return {
        "summary": profile.summary,
        "explicit_info": [i.model_dump() for i in profile.explicit_info],
        "implicit_traits": [i.model_dump() for i in profile.implicit_traits],
        "profile_timestamp_ms": profile.timestamp,
    }
