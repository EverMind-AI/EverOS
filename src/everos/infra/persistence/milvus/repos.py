"""Milvus repositories for EverOS derived index tables, one per table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from everos.core.utils.datetime import from_timestamp_ms
from everos.infra.persistence.records import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    UserProfile,
)
from everos.memory.ports.query import all_of, eq, gt, is_null

from .repository import MilvusRepoBase


class _EpisodeRepo(MilvusRepoBase[Episode]):
    schema = Episode

    async def count_by_owner(
        self,
        owner_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
        parent_type: str | None = None,
    ) -> int:
        return await self._count_where(
            all_of(
                eq("owner_id", owner_id),
                eq("app_id", app_id),
                eq("project_id", project_id),
                is_null("deprecated_by"),
                eq("parent_type", parent_type) if parent_type is not None else None,
            )
        )

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
        predicate = all_of(
            eq("owner_id", owner_id),
            gt("timestamp", from_timestamp_ms(after_ts)),
            eq("parent_type", parent_type),
            eq("app_id", app_id),
            eq("project_id", project_id),
            is_null("deprecated_by"),
        )
        # A plain query(limit=...) cannot cross Milvus' 16,384-row window, so
        # the ceiling has to be applied through the iterator instead.
        raw = await self._scan_raw(
            predicate, include_vectors=True, max_rows=limit or 20_000
        )
        rows = [self._codec.to_model(row) for row in raw]
        rows.sort(key=lambda row: row.timestamp)
        if columns is None:
            return rows
        projection = list(dict.fromkeys([*columns, "timestamp"]))
        return [{name: getattr(row, name) for name in projection} for row in rows]


class _ImageMemoryRepo(MilvusRepoBase[ImageMemory]):
    schema = ImageMemory


class _GroupProfileItemRepo(MilvusRepoBase[GroupProfileItem]):
    schema = GroupProfileItem


class _UserProfileRepo(MilvusRepoBase[UserProfile]):
    schema = UserProfile
