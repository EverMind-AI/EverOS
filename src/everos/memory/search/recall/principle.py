"""Principle recall — KV-by-owner LanceDB fetch (no ranking).

Principle is Meta Memory, not a product Kind: there is no HYBRID /
BM25 / vector lane and ``kinds`` never accepts ``"principle"``. The
recaller is a deliberate KV lookup: given ``owner_id`` + app/project
scope, return every ``principle`` row for that user (one file exploded
to N rows). There is no ``query`` and no ``score``.

The cascade keeps ``Principle`` rows in sync with
``users/<user_id>/principles.md``; this recaller just reads them.
Unlike :class:`ProfileRecaller` (PK ``id = owner_id``, at most one
row), principle PKs are ``<owner_id>_<principle_id>`` so the fetch
filters ``owner_id`` **and** ``app_id`` **and** ``project_id``.
"""

from __future__ import annotations

from everos.component.utils.datetime import from_timestamp, to_display_tz
from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb import principle_repo

from ..dto import SearchPrincipleItem

logger = get_logger(__name__)

_FETCH_LIMIT = 1000


class PrincipleRecaller:
    """Fetch the owner's principle rows from LanceDB (N rows, not 1)."""

    async def fetch(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[SearchPrincipleItem]:
        """Return the owner's principle items, or ``[]`` when none exist.

        Empty list (rather than 404) lets the caller emit a normal
        response with ``principles=[]`` while the user has no synthesised
        principles yet.
        """
        if not owner_id:
            return []
        where = (
            f"owner_id = '{_q(owner_id)}' AND "
            f"app_id = '{_q(app_id)}' AND "
            f"project_id = '{_q(project_id)}'"
        )
        rows = await principle_repo.find_where(where, limit=_FETCH_LIMIT)
        if not rows:
            logger.debug("principle_fetch_miss", owner_id=owner_id)
            return []
        items: list[SearchPrincipleItem] = []
        for row in rows:
            ts = from_timestamp(row.timestamp_ms)
            items.append(
                SearchPrincipleItem(
                    id=row.id,
                    user_id=row.owner_id,
                    app_id=row.app_id,
                    project_id=row.project_id,
                    title=row.title,
                    statement=row.statement,
                    source_entry_ids=list(row.source_entry_ids),
                    timestamp=to_display_tz(ts) or ts,
                    score=None,
                )
            )
        return items


def _q(value: str) -> str:
    """Escape single quotes for a LanceDB SQL-like ``where`` predicate."""
    return value.replace("'", "''")
