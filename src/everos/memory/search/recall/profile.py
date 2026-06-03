"""Profile recall — KV-by-owner LanceDB fetch (no ranking).

Profile is the only owner-scoped kind that ships as **one row per
user** (no per-day fan-out, no entry markers). The recaller is a
deliberate KV-by-owner lookup: given ``owner_id``, return at most one
:class:`SearchProfileItem`. There is no ``query`` and no ``score``
field on the response — the DTO's optional ``score`` is reserved for
a future query-aware lookup.

The cascade keeps ``UserProfile`` rows in sync with
``users/<user_id>/user.md``; this recaller just reads the row and
unpacks the json-encoded buckets back into the DTO's
``profile_data`` mapping (mirrors enterprise's profile DTO shape).
"""

from __future__ import annotations

import json
from typing import Any

from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb import user_profile_repo

from ..dto import SearchProfileItem

logger = get_logger(__name__)


class ProfileRecaller:
    """Fetch the owner's profile row from LanceDB, return at most one item."""

    async def fetch(self, owner_id: str) -> list[SearchProfileItem]:
        """Return ``[item]`` if a profile row exists, otherwise ``[]``.

        Empty list (rather than 404) lets the caller emit a normal
        response with ``profiles=[]`` while the user is still in their
        cold-start window (no profile synthesised yet).
        """
        if not owner_id:
            return []
        row = await user_profile_repo.get_by_id(owner_id)
        if row is None:
            logger.debug("profile_fetch_miss", owner_id=owner_id)
            return []
        profile_data: dict[str, Any] = {
            "summary": row.summary,
            "explicit_info": _load_json(row.explicit_info_json),
            "implicit_traits": _load_json(row.implicit_traits_json),
            "profile_timestamp_ms": row.profile_timestamp_ms,
        }
        return [
            SearchProfileItem(
                id=row.id,
                user_id=row.owner_id,
                app_id=row.app_id,
                project_id=row.project_id,
                profile_data=profile_data,
                score=None,
            )
        ]


def _load_json(text: str) -> Any:
    """Decode a json-encoded frontmatter bucket.

    Returns ``[]`` on empty / malformed input so a row with a stale
    encoding doesn't blow up the search response. A real decode error
    is logged once at debug; cascade will rewrite the column on the
    next reconcile.
    """
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("profile_json_decode_failed", payload_head=text[:80])
        return []
