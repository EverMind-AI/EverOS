"""LanceDB repo singleton for the ``episode`` table."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceDailyLogRepoBase
from everos.core.persistence.lancedb.repository import _q

from ..lancedb_manager import get_table
from ..tables.episode import Episode


class _EpisodeRepo(LanceDailyLogRepoBase[Episode]):
    schema = Episode

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)

    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int:
        """Count episode rows for one owner within an ``(app, project)`` scope.

        Args:
            owner_id: The row's ``owner_id`` field.
            app_id: App scope segment (default ``"default"``).
            project_id: Project scope segment (default ``"default"``).
            parent_type: When non-``None``, restrict the count to rows whose
                ``parent_type`` matches (e.g. ``"memcell"`` to exclude
                Reflection-merged rows with ``parent_type='cluster'``). Kept
                optional so callers without a parent-type requirement rely on
                the ``None`` default meaning "count everything".

        Used by :func:`extract_user_profile`'s Tier-1 direct-path throttle
        to gate re-extraction frequency. LanceDB ``count_rows`` accepts a
        SQL-style ``filter`` predicate; the where-string is built with the
        same escape discipline (:func:`_q`) as
        :meth:`list_by_owner_after_ts`.

        Excludes rows with ``deprecated_by IS NOT NULL`` (Reflection-
        superseded episodes) from the count, matching the filter idiom
        used by :func:`memory.search.filters.compile_filters` and the
        reflection read paths. Keeps the profile-throttle count aligned
        with the "live" episode surface the strategy actually operates on.
        """
        filter_str = (
            f"owner_id = '{_q(owner_id)}' "
            f"AND app_id = '{_q(app_id)}' "
            f"AND project_id = '{_q(project_id)}' "
            f"AND deprecated_by IS NULL"
        )
        if parent_type is not None:
            filter_str += f" AND parent_type = '{_q(parent_type)}'"
        table = await self._table()
        return await table.count_rows(filter=filter_str)

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
        """Scalar filter over owner + timestamp + parent_type + scope.

        Fetches episodes for the given owner after a specific timestamp,
        scoped by parent_type and app/project. No vector dependency —
        Tier 1-safe. Rows with vector=None are included.

        Used by extract_user_profile direct path (KEYWORD_ONLY mode) to
        select memcells to feed the profile extractor without going through
        cluster events.

        Excludes rows with ``deprecated_by IS NOT NULL`` (Reflection-
        superseded episodes) — matches the filter idiom used by
        :func:`memory.search.filters.compile_filters` and reflection
        reads. Without this, direct-path profile extraction would feed
        stale (superseded) memcell references to the extractor.

        Args:
            owner_id: Owner identifier.
            after_ts: Timestamp threshold (milliseconds since epoch); rows with
                timestamp > after_ts are returned.
            parent_type: Parent type filter (e.g. "memcell").
            app_id: Application scope (default "default").
            project_id: Project scope (default "default").
            columns: Optional column projection. Default (``None``)
                returns full ``Episode`` model objects. Callers that only
                need a small subset (e.g. ``parent_id`` in
                ``extract_user_profile``) should pass an explicit list to
                skip the 1024-D ``vector`` and ``subject_vector`` payload.
                When set, this method returns raw dicts scoped to the
                requested columns; otherwise it returns typed ``Episode``
                objects. Callers switch on the projection they asked for.
            limit: Optional row cap. Default (``None``) means no cap —
                the entire matching set is fetched. Callers with large
                historical windows should pass a bound to avoid pulling
                every memcell for the owner.

                **CAVEAT — limit is NOT "newest N"**: LanceDB does not
                push ORDER BY down; ``.limit(N)`` truncates whatever
                the scan returns first (typically fragment / insertion
                order for an append-only table = OLDEST N). The
                ascending-by-timestamp sort below runs on the already-
                truncated Python list, so the returned rows are the
                *oldest matching subset*, not the newest. Callers that
                need "most recent N" MUST either fetch unlimited and
                slice in Python (accepting the memory cost), or apply
                a tighter ``after_ts`` bound to shrink the matching
                set upstream. A regression here would silently window
                profile extraction over the wrong slice of history.

        Returns:
            When ``columns`` is ``None``: list of ``Episode`` objects
            ordered ascending by timestamp (existing behaviour).
            When ``columns`` is a projection: list of raw dicts holding
            only the requested columns, ordered ascending by timestamp.
        """
        from everos.component.utils.datetime import to_iso_format

        # Convert milliseconds to UTC datetime, then to ISO string for DataFusion
        # (stored timestamps are always UTC)
        ts_dt = dt.datetime.fromtimestamp(int(after_ts) / 1000.0, tz=dt.UTC)
        ts_iso = to_iso_format(ts_dt)

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
        # Projection: caller opted into a raw-dict return by naming columns;
        # `timestamp` is force-included so the ascending sort below is stable
        # even when the caller forgot to list it.
        projected = columns is not None
        if projected:
            projection = list(dict.fromkeys([*columns, "timestamp"]))  # type: ignore[misc]
            query = query.select(projection)
        if limit is not None:
            query = query.limit(limit)
        rows = await query.to_list()
        rows.sort(key=lambda r: r["timestamp"])
        if projected:
            return rows
        return [self.schema.model_validate(r) for r in rows]


episode_repo = _EpisodeRepo()
