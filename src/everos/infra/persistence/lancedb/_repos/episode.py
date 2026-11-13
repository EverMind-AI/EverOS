"""LanceDB repository for the ``episode`` table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from everos.core.utils.datetime import ensure_utc, from_timestamp_ms, to_iso_format
from everos.infra.persistence.lancedb._base.repository import LanceDailyLogRepoBase, _q

from ...records.episode import Episode
from ._contract import LanceIndexContract


class _EpisodeRepo(LanceIndexContract[Episode], LanceDailyLogRepoBase[Episode]):
    schema = Episode

    async def list_by_owner_after_ts(
        self,
        *,
        owner_id: str,
        after_ts: int,
        parent_type: str,
        app_id: str = "default",
        project_id: str = "default",
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[Episode] | list[dict[str, Any]]:
        """One owner's live episodes observed after ``after_ts``.

        A scalar filter over owner + timestamp + parent_type + scope, so it
        needs no vectors and rows without one are included. Rows superseded by
        Reflection (``deprecated_by`` set) are left out, as in every other
        episode read.

        Args:
            owner_id: Owner identifier.
            after_ts: Epoch milliseconds; rows with ``timestamp > after_ts``
                are returned.
            parent_type: Parent type filter (e.g. ``"memcell"``).
            app_id: Application scope.
            project_id: Project scope.
            columns: Column projection. ``None`` returns ``Episode`` objects;
                a list returns raw dicts with only those columns (plus
                ``timestamp``), so a caller can leave the vectors behind.
            limit: Row cap; ``None`` fetches every match. LanceDB does not
                push ``ORDER BY`` down, so the cap keeps whichever rows the
                scan yields first (insertion order for this append-only
                table), not the newest; the timestamp sort runs afterwards.

        Returns:
            The rows, ascending by timestamp.
        """
        ts_iso = to_iso_format(ensure_utc(from_timestamp_ms(after_ts)))
        where = (
            f"owner_id = '{_q(owner_id)}' "
            f"AND timestamp > TIMESTAMP '{ts_iso}' "
            f"AND parent_type = '{_q(parent_type)}' "
            f"AND app_id = '{_q(app_id)}' "
            f"AND project_id = '{_q(project_id)}' "
            f"AND deprecated_by IS NULL"
        )
        table = await self._table()
        query = table.query().where(where)
        projected = columns is not None
        if projected:
            # ``timestamp`` rides along so the sort below never misses its key.
            projection = list(dict.fromkeys([*columns, "timestamp"]))  # type: ignore[misc]
            query = query.select(projection)
        if limit is not None:
            query = query.limit(limit)
        rows = await query.to_list()
        rows.sort(key=lambda r: r["timestamp"])
        if projected:
            return rows
        return [self.schema.model_validate(r) for r in rows]
