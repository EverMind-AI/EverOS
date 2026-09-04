"""Profile recall — KV-by-owner LanceDB fetch (no ranking).

Profile has no per-day fan-out and no entry markers. The recaller is a
deliberate KV-by-owner lookup — there is no ``query`` and no ``score`` on the
response; the DTO's optional ``score`` is reserved for a future query-aware
lookup.

One owner usually holds exactly one row (``subject`` empty, itself the
subject). A **group** owner holds one row per participant, so the fetch takes
an optional ``subject``: naming one returns that person's profile, omitting it
returns every row under the owner. ``PROFILE_MAX_ROWS`` bounds the second case
— a caller that wants one person out of a large group has to say which,
because silently truncating would answer as somebody else.

The cascade keeps ``UserProfile`` rows in sync with the md; this recaller just
reads them and unpacks the json-encoded buckets back into the DTO's
``profile_data`` mapping (mirrors enterprise's profile DTO shape).
"""

from __future__ import annotations

import json
from typing import Any

from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb import user_profile_repo

from ..dto import SearchProfileItem

logger = get_logger(__name__)

PROFILE_MAX_ROWS = 64
"""Cap on an unfiltered group fetch. Above this the response is a wall of
other people's profiles; the caller should pass ``subject`` instead."""


class ProfileRecaller:
    """Fetch an owner's profile rows from LanceDB."""

    async def fetch(
        self, owner_id: str, *, subject: str | None = None
    ) -> list[SearchProfileItem]:
        """Return the owner's profile rows, or ``[]`` when there are none.

        Empty list (rather than 404) lets the caller emit a normal
        response with ``profiles=[]`` while the user is still in their
        cold-start window (no profile synthesised yet).

        Args:
            owner_id: Memory partition. Also the row id in the common
                owner-is-subject case.
            subject: Name of one participant of a group owner. ``None``
                returns every row under the owner, ordered by subject so the
                response is stable across calls.
        """
        if not owner_id:
            return []
        if subject:
            row = await user_profile_repo.get_by_id(f"{owner_id}::{subject}")
            rows = [row] if row is not None else []
        else:
            rows = sorted(
                await user_profile_repo.find_by_owner(owner_id, limit=PROFILE_MAX_ROWS),
                key=lambda r: r.id,
            )
        if not rows:
            logger.debug("profile_fetch_miss", owner_id=owner_id, subject=subject)
            return []
        if len(rows) == PROFILE_MAX_ROWS:
            logger.warning(
                "profile_fetch_truncated",
                owner_id=owner_id,
                limit=PROFILE_MAX_ROWS,
            )
        return [
            SearchProfileItem(
                id=row.id,
                user_id=row.owner_id,
                app_id=row.app_id,
                project_id=row.project_id,
                profile_data={
                    "subject": _subject_of(row.id, row.owner_id) or row.owner_id,
                    "summary": row.summary,
                    "explicit_info": _load_json(row.explicit_info_json),
                    "implicit_traits": _load_json(row.implicit_traits_json),
                    "profile_timestamp_ms": row.profile_timestamp_ms,
                },
                score=None,
            )
            for row in rows
        ]


def _subject_of(row_id: str, owner_id: str) -> str:
    """Recover the subject the cascade encoded into the row id.

    ``<owner_id>::<subject>`` for a group owner's participant; a bare
    ``owner_id`` means the owner is its own subject, so the subject is empty.
    Stripping the known prefix (rather than splitting on ``::``) keeps this
    unambiguous whatever the name contains.
    """
    prefix = f"{owner_id}::"
    return row_id[len(prefix) :] if row_id.startswith(prefix) else ""


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
