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
                profile_data={
                    "summary": profile.summary,
                    "explicit_info": profile.explicit_info,
                    "implicit_traits": profile.implicit_traits,
                    "profile_timestamp_ms": profile.timestamp,
                },
                score=None,
            )
        ]
