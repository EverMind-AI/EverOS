"""SQLite adapters for the memory domain's state ports.

Each ``Sqlite*Store`` wraps one of the table repositories and owns the
row <-> domain conversion, so row models never cross into the domain.
Session-state tracks are fixed here: callers address a conversation by
:class:`~everos.memory.ports.SessionKey` and never see a track name.
"""

from __future__ import annotations

__all__ = [
    "SqliteClusterStore",
    "SqliteMemcellStore",
    "SqliteMemorizeInboxStore",
    "SqliteReflectionReportStore",
    "SqliteSessionBufferStore",
]

import datetime as dt
import json
from collections.abc import Collection, Sequence
from typing import cast

import numpy as np
from everalgo.types import MemCell

from everos.component.utils.datetime import from_timestamp
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import (
    CanonicalMessage,
    Cluster,
    ClusterKind,
    ClusterMemberType,
    MemcellRecord,
    MemorizeTask,
    ReflectionReport,
    ToolCall,
)
from everos.memory.ports import OwnerType, PageResult, Scope, SessionKey

from .repos import (
    ClusterRepo,
    ConversationStatusRepo,
    MemcellRepo,
    MemorizeInboxRepo,
    ReflectionReportRepo,
    UnprocessedBufferRepo,
)
from .tables import Memcell, MemorizeInbox, UnprocessedBuffer
from .tables import ReflectionReport as ReflectionReportRow

_SESSION_TRACK = "memorize"
"""Track of a session's live buffer and its conversation status."""
_QUARANTINE_TRACK = "quarantine"
"""Track a buffer is moved to when it cannot be segmented."""
_CENTROID_DTYPE = np.float32


class SqliteSessionBufferStore:
    """Session buffer + conversation status over ``unprocessed_buffer``.

    Args:
        buffer_repo: Repository of the ``unprocessed_buffer`` table.
        status_repo: Repository of the ``conversation_status`` table.
    """

    def __init__(
        self,
        buffer_repo: UnprocessedBufferRepo,
        status_repo: ConversationStatusRepo,
    ) -> None:
        self._buffer = buffer_repo
        self._status = status_repo

    async def list(self, session: SessionKey) -> list[CanonicalMessage]:
        rows = await self._buffer.list_for_track(
            session.session_id,
            _SESSION_TRACK,
            app_id=session.app_id,
            project_id=session.project_id,
        )
        return [_row_to_message(row) for row in rows]

    async def replace(
        self, session: SessionKey, messages: Sequence[CanonicalMessage]
    ) -> None:
        await self._buffer.replace(
            session.session_id,
            _SESSION_TRACK,
            [_message_to_row(m, session) for m in messages],
            app_id=session.app_id,
            project_id=session.project_id,
        )

    async def quarantine(
        self, session: SessionKey, extra: Sequence[CanonicalMessage]
    ) -> int:
        return await self._buffer.quarantine(
            session.session_id,
            _SESSION_TRACK,
            _QUARANTINE_TRACK,
            [_message_to_row(m, session) for m in extra],
            app_id=session.app_id,
            project_id=session.project_id,
        )

    async def touch_last_message(self, session: SessionKey, ts: dt.datetime) -> None:
        await self._status.touch_last_message_ts(
            session.session_id,
            _SESSION_TRACK,
            ts,
            app_id=session.app_id,
            project_id=session.project_id,
        )

    async def touch_last_memcell(self, session: SessionKey, ts: dt.datetime) -> None:
        await self._status.touch_last_memcell_ts(
            session.session_id,
            _SESSION_TRACK,
            ts,
            app_id=session.app_id,
            project_id=session.project_id,
        )


class SqliteMemcellStore:
    """Memcells over the ``memcell`` table.

    Args:
        repo: Repository of the ``memcell`` table.
    """

    def __init__(self, repo: MemcellRepo) -> None:
        self._repo = repo

    async def insert_many(
        self, session: SessionKey, records: Sequence[MemcellRecord]
    ) -> None:
        await self._repo.insert_many([_record_to_row(r, session) for r in records])

    async def find_by_ids(self, memcell_ids: Sequence[str]) -> list[MemCell]:
        rows = await self._repo.find_by_ids(list(memcell_ids))
        return [MemCell.model_validate_json(row.payload_json) for row in rows]


class SqliteClusterStore:
    """Clusters and membership over ``cluster`` + ``cluster_member``.

    Args:
        repo: Repository of the cluster table pair.
    """

    def __init__(self, repo: ClusterRepo) -> None:
        self._repo = repo

    async def list_for_owner(self, scope: Scope, kind: ClusterKind) -> list[Cluster]:
        clusters = await self._repo.list_for_owner(
            scope.owner_id, kind, app_id=scope.app_id, project_id=scope.project_id
        )
        return [cluster_mapper.from_algo(c) for c in clusters]

    async def get(self, cluster_id: str) -> Cluster | None:
        found = await self._repo.get_with_members(cluster_id)
        return None if found is None else cluster_mapper.from_algo(found)

    async def upsert(
        self,
        scope: Scope,
        kind: ClusterKind,
        cluster: Cluster,
        *,
        member_type: ClusterMemberType,
    ) -> None:
        await self._repo.upsert_with_members(
            cluster_mapper.to_algo(cluster),
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
            kind=kind,
            member_type=member_type,
            app_id=scope.app_id,
            project_id=scope.project_id,
        )

    async def find_cluster_for_member(
        self, scope: Scope, member_type: ClusterMemberType, member_id: str
    ) -> str | None:
        return await self._repo.find_cluster_id_for_member(
            member_type,
            member_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
        )

    async def members(self, cluster_id: str) -> list[tuple[str, str]]:
        return [
            (member_id, member_type)
            for member_id, member_type in await self._repo.get_members_with_type(
                cluster_id
            )
        ]

    async def add_member(
        self, cluster_id: str, member_id: str, member_type: ClusterMemberType
    ) -> None:
        await self._repo.add_member(cluster_id, member_id, member_type)

    async def remove_members(
        self, cluster_id: str, member_ids: Collection[str]
    ) -> None:
        await self._repo.remove_members(cluster_id, set(member_ids))

    async def update_metadata(
        self,
        cluster_id: str,
        *,
        centroid: Sequence[float],
        count: int,
        last_ts: int,
        preview: Sequence[str],
    ) -> None:
        await self._repo.update_metadata(
            cluster_id,
            centroid_blob=np.asarray(centroid, dtype=_CENTROID_DTYPE).tobytes(),
            count=count,
            last_ts_ms=last_ts,
            preview_json=json.dumps(list(preview), ensure_ascii=False),
        )

    async def list_ids_and_member_counts(
        self, scope: Scope, kind: ClusterKind
    ) -> list[tuple[str, int]]:
        return [
            (cluster_id, int(count))
            for cluster_id, count in await self._repo.list_ids_and_member_counts(
                scope.owner_id,
                kind,
                app_id=scope.app_id,
                project_id=scope.project_id,
            )
        ]

    async def list_distinct_owners(self) -> list[Scope]:
        return [
            Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=owner_id,
                owner_type=cast(OwnerType, owner_type),
            )
            for owner_id, owner_type, app_id, project_id in (
                await self._repo.list_distinct_owners()
            )
        ]

    async def count(self) -> int:
        return await self._repo.count()


class SqliteMemorizeInboxStore:
    """The asynchronous memorize queue over ``memorize_inbox``.

    Args:
        repo: Repository of the ``memorize_inbox`` table.
    """

    def __init__(self, repo: MemorizeInboxRepo) -> None:
        self._repo = repo

    async def enqueue(
        self,
        session: SessionKey,
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
        payload_json: str,
        message_count: int,
    ) -> MemorizeTask:
        row = await self._repo.add(
            MemorizeInbox(
                request_id=request_id,
                app_id=session.app_id,
                project_id=session.project_id,
                session_id=session.session_id,
                is_final=is_final,
                defer_extraction=defer_extraction,
                payload_json=payload_json,
                message_count=message_count,
            )
        )
        return _row_to_task(row)

    async def claim_next(
        self, *, exclude_sessions: Collection[SessionKey] = ()
    ) -> MemorizeTask | None:
        row = await self._repo.claim_next(
            exclude_sessions=[
                (key.app_id, key.project_id, key.session_id) for key in exclude_sessions
            ]
        )
        return None if row is None else _row_to_task(row)

    async def mark(
        self,
        task_id: int,
        *,
        status: str,
        attempt: int | None = None,
        last_error: str | None = None,
        payload_json: str | None = None,
    ) -> None:
        changes = {
            name: value
            for name, value in (
                ("attempt", attempt),
                ("last_error", last_error),
                ("payload_json", payload_json),
            )
            if value is not None
        }
        await self._repo.mark(task_id, status=status, **changes)

    async def count_pending(self, session: SessionKey | None = None) -> int:
        if session is None:
            return await self._repo.count_pending()
        return await self._repo.count_pending(
            session_id=session.session_id,
            app_id=session.app_id,
            project_id=session.project_id,
        )

    async def count_status(self, status: str) -> int:
        return await self._repo.count_status(status)

    async def fail_abandoned(self, *, exclude_ids: Collection[int] = ()) -> int:
        return await self._repo.fail_abandoned(exclude_ids=exclude_ids)

    async def requeue_processing(self, *, max_replays: int = 3) -> tuple[int, int]:
        return await self._repo.requeue_processing(max_replays=max_replays)

    async def oldest_pending_at(self) -> dt.datetime | None:
        return await self._repo.oldest_pending_created_at()

    async def list_tasks(
        self,
        *,
        session: SessionKey | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> PageResult[MemorizeTask]:
        rows, total = await self._repo.list_tasks(
            session_id=None if session is None else session.session_id,
            status=status,
            app_id="default" if session is None else session.app_id,
            project_id="default" if session is None else session.project_id,
            offset=offset,
            limit=limit,
        )
        return PageResult(items=[_row_to_task(r) for r in rows], total=total)

    async def get_task(self, request_id: str) -> MemorizeTask | None:
        row = await self._repo.find_one(request_id=request_id)
        return None if row is None else _row_to_task(row)

    async def purge_expired(self, *, before: dt.datetime) -> tuple[int, int]:
        return await self._repo.purge_expired(before=before)


class SqliteReflectionReportStore:
    """Reflection audit trail over ``reflection_report``.

    Args:
        repo: Repository of the ``reflection_report`` table.
    """

    def __init__(self, repo: ReflectionReportRepo) -> None:
        self._repo = repo

    async def create(self, report: ReflectionReport) -> None:
        await self._repo.create(_report_to_row(report))

    async def reflected_cluster_ids(self, scope: Scope) -> set[str]:
        return await self._repo.list_reflected_cluster_ids(
            scope.owner_id, scope.app_id, scope.project_id
        )


# ── Row <-> domain ───────────────────────────────────────────────────────


def _message_to_row(m: CanonicalMessage, session: SessionKey) -> UnprocessedBuffer:
    return UnprocessedBuffer(
        message_id=m.message_id,
        app_id=session.app_id,
        project_id=session.project_id,
        session_id=m.session_id,
        track=_SESSION_TRACK,
        sender_id=m.sender_id,
        sender_name=m.sender_name,
        role=m.role,
        timestamp=m.timestamp,
        content_items_json=json.dumps(m.content_items),
        text=m.text,
        tool_calls_json=(
            json.dumps([tc.model_dump() for tc in m.tool_calls])
            if m.tool_calls
            else None
        ),
        tool_call_id=m.tool_call_id,
    )


def _row_to_message(row: UnprocessedBuffer) -> CanonicalMessage:
    tool_calls: list[ToolCall] | None = None
    if row.tool_calls_json:
        tool_calls = [
            ToolCall.model_validate(d) for d in json.loads(row.tool_calls_json)
        ]
    content_items = json.loads(row.content_items_json) if row.content_items_json else []
    # ``row.timestamp`` is UtcDatetime: the table load hook re-attaches UTC.
    return CanonicalMessage(
        message_id=row.message_id,
        session_id=row.session_id,
        sender_id=row.sender_id,
        sender_name=row.sender_name,
        role=row.role,  # type: ignore[arg-type]  # stored from the same Literal
        timestamp=row.timestamp,
        content_items=content_items,
        text=row.text,
        tool_calls=tool_calls,
        tool_call_id=row.tool_call_id,
    )


def _record_to_row(record: MemcellRecord, session: SessionKey) -> Memcell:
    return Memcell(
        memcell_id=record.memcell_id,
        app_id=session.app_id,
        project_id=session.project_id,
        session_id=session.session_id,
        track=_SESSION_TRACK,
        raw_type=record.raw_type,
        message_ids_json=json.dumps(record.message_ids),
        sender_ids_json=json.dumps(record.sender_ids),
        payload_json=record.cell.model_dump_json(),
        timestamp=from_timestamp(record.cell.timestamp),
    )


def _row_to_task(row: MemorizeInbox) -> MemorizeTask:
    assert row.id is not None  # set by the database on insert
    return MemorizeTask(
        id=row.id,
        request_id=row.request_id,
        app_id=row.app_id,
        project_id=row.project_id,
        session_id=row.session_id,
        is_final=row.is_final,
        defer_extraction=row.defer_extraction,
        payload_json=row.payload_json,
        message_count=row.message_count,
        status=row.status,
        attempt=row.attempt,
        last_error=row.last_error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _report_to_row(report: ReflectionReport) -> ReflectionReportRow:
    return ReflectionReportRow(
        id=report.id,
        cluster_id=report.cluster_id,
        owner_id=report.owner_id,
        app_id=report.app_id,
        project_id=report.project_id,
        mode=report.mode,
        source_members=json.dumps(
            [
                {"member_id": member_id, "member_type": member_type}
                for member_id, member_type in report.source_members
            ],
            ensure_ascii=False,
        ),
        source_count=report.source_count,
        merged_entry_id=report.merged_entry_id,
        deprecated_fact_count=report.deprecated_fact_count,
    )
