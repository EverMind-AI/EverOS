"""Backend-neutral adapters from the derived-index repositories to the ports.

Both engines expose the same :class:`IndexRepository` contract, so one
adapter per memory kind serves either;
:func:`everos.infra.persistence.factories.build_indexes` binds each to the
selected engine's repository. The adapter
owns the row -> domain conversion and the per-kind query details the
domain must not know (BM25 columns, vector column names, projections).
"""

from __future__ import annotations

__all__ = [
    "EpisodeIndexAdapter",
    "GroupProfileIndexAdapter",
    "ImageMemoryIndexAdapter",
    "ProfileIndexAdapter",
]

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import to_timestamp_ms
from everos.memory.models import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    Profile,
)
from everos.memory.ports import (
    Filter,
    MemoryRef,
    OwnerType,
    Page,
    PageResult,
    Scope,
    SearchHit,
    Stored,
    primary_key,
)

from ..records import member_profile_row_id
from .protocols import IndexRepository

logger = get_logger(__name__)

type Row = Mapping[str, Any]


class _SearchableIndex[T]:
    """BM25 + ANN + filtered scan shared by every owner-scoped searchable kind.

    Args:
        repo: The kind's derived-index repository.
        to_stored: The kind's row -> domain conversion.
    """

    def __init__(
        self, repo: IndexRepository[Any], to_stored: Callable[[Row], Stored[T]]
    ) -> None:
        self._repo = repo
        self._to_stored = to_stored

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[T]]:
        rows = await self._repo.sparse_search(
            terms, where, columns=self._repo.schema.BM25_FIELDS, limit=limit
        )
        return [self._hit(row, _bm25_score(row)) for row in rows]

    async def dense_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[T]]:
        rows = await self._repo.dense_search(vector, where, limit=limit)
        return [self._hit(row, _cosine_similarity(row)) for row in rows]

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[SearchHit[T]]:
        if limit is None:
            rows: Sequence[BaseModel | Row] = await self._repo.scan(where)
        else:
            rows = await self._repo.search(where=where, limit=limit)
        return [self._hit(row, None) for row in rows]

    def _hit(self, row: BaseModel | Row, score: float | None) -> SearchHit[T]:
        return SearchHit(stored=self._to_stored(_fields(row)), score=score)

    async def _paginate(self, where: Filter, page: Page) -> PageResult[Stored[T]]:
        rows, total = await self._repo.find_where_paginated(
            where,
            sort_by=page.sort_by,
            descending=page.descending,
            page=page.number,
            page_size=page.size,
        )
        return PageResult(
            items=[self._to_stored(_fields(r)) for r in rows], total=total
        )


class EpisodeIndexAdapter(_SearchableIndex[Episode]):
    """Episode index reads.

    Args:
        repo: Episode derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        super().__init__(repo, _stored_episode)

    async def dense_search_subject(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[Episode]]:
        rows = await self._repo.dense_search(
            vector, where, limit=limit, vector_field="subject_vector"
        )
        return [self._hit(row, _cosine_similarity(row)) for row in rows]

    async def paginate(self, where: Filter, page: Page) -> PageResult[Stored[Episode]]:
        return await self._paginate(where, page)


class ImageMemoryIndexAdapter(_SearchableIndex[ImageMemory]):
    """Picture-memory index reads — one route more than an episode gets.

    ``dense_search`` scores ``vector`` (the description's text embedding);
    :meth:`visual_search` scores ``visual_vector`` (the picture bytes' own
    embedding). The two spaces are different models, so the caller fuses the
    two result lists by rank.

    Args:
        repo: Image-memory derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        super().__init__(repo, _stored_image_memory)

    async def visual_search(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> list[SearchHit[ImageMemory]]:
        """ANN over the picture's own embedding; empty when none is stored."""
        if not vector:
            return []
        rows = await self._repo.dense_search(
            vector, where, limit=limit, vector_field="visual_vector"
        )
        return [self._hit(row, _cosine_similarity(row)) for row in rows]

    async def paginate(
        self, where: Filter, page: Page
    ) -> PageResult[Stored[ImageMemory]]:
        return await self._paginate(where, page)


class ProfileIndexAdapter:
    """The one profile row per user, and per member inside a group.

    A member-in-group profile row is ``owner_type = group`` with the member
    in ``subject_id`` (docs/group-memory.md, section 6.3); its row id can
    never equal a user id, and :meth:`get` answers user rows only.

    Args:
        repo: User-profile derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        self._repo = repo

    async def get(self, scope: Scope) -> Stored[Profile] | None:
        row = await self._repo.find_one_where(
            primary_key(scope.app_id, scope.project_id, scope.owner_id)
        )
        if row is None:
            return None
        fields = _fields(row)
        if fields["owner_type"] != "user":
            return None
        return _stored_profile(fields, subject=fields["owner_id"])

    async def member(self, scope: Scope, member_id: str) -> Stored[Profile] | None:
        row = await self._repo.find_one_where(
            primary_key(
                scope.app_id,
                scope.project_id,
                member_profile_row_id(scope.owner_id, member_id),
            )
        )
        if row is None:
            return None
        fields = _fields(row)
        if fields["owner_type"] != "group" or fields.get("subject_id") != member_id:
            return None
        return _stored_profile(fields, subject=member_id)


class GroupProfileIndexAdapter:
    """The group-profile items in the derived index (``group_profile_item``).

    Filters name the row's columns, which carry the domain's own names
    (``kind``, ``topic_id``, ``supersedes``, ``about``, ``member_id``,
    ``source_memcell_ids``, ``subject_ids``, ``observed_at`` ...) plus the
    owner and window columns every group record has, so a read scope's
    predicate applies unchanged; time columns compare with datetimes, as the
    scope predicate's window bounds do. A row's version state is not filtered
    here: ``is_null("deprecated_by")`` keeps the newest versions only, and
    without it superseded and forgotten rows come back marked as such.

    Each item is rebuilt from the row's ``payload_json`` with ``item_id`` (its
    entry id) and the ``status`` / ``superseded_by`` its ``deprecated_by``
    gives it. A row whose payload no longer validates is logged and skipped:
    a stale encoding must not fail a read, and the cascade rewrites the row
    on its next reconcile of the file.

    Args:
        repo: Group-profile derived-index repository.
    """

    def __init__(self, repo: IndexRepository[Any]) -> None:
        self._repo = repo

    async def find(
        self, where: Filter, *, limit: int | None = None
    ) -> list[Stored[GroupProfileItem]]:
        rows = [_fields(row) for row in await self._repo.scan(where)]
        rows.sort(key=lambda row: (row["observed_at"], row["entry_id"]), reverse=True)
        found = _stored_group_profile_items(rows)
        return found if limit is None else found[:limit]

    async def sparse_search(
        self, terms: Sequence[str], where: Filter, *, limit: int
    ) -> list[SearchHit[GroupProfileItem]]:
        rows = await self._repo.sparse_search(
            terms, where, columns=self._repo.schema.BM25_FIELDS, limit=limit
        )
        hits: list[SearchHit[GroupProfileItem]] = []
        for row in rows:
            stored = _stored_group_profile_item(row)
            if stored is not None:
                hits.append(SearchHit(stored=stored, score=_bm25_score(row)))
        return hits


# ── Row -> domain ────────────────────────────────────────────────────────


def _fields(row: BaseModel | Row) -> Row:
    return row.model_dump(mode="python") if isinstance(row, BaseModel) else row


def _bm25_score(row: Row) -> float | None:
    score = row.get("_score")
    return None if score is None else float(score)


def _cosine_similarity(row: Row) -> float:
    """Both engines report ``_distance = 1 - cos``; a missing one scores 0."""
    distance = row.get("_distance")
    if distance is None:
        return 0.0
    return min(max(1.0 - float(distance), 0.0), 1.0)


def _scope(fields: Row) -> Scope:
    return Scope(
        app_id=fields["app_id"],
        project_id=fields["project_id"],
        owner_id=fields["owner_id"],
        owner_type=cast(OwnerType, fields["owner_type"]),
    )


def _daily_ref(fields: Row) -> MemoryRef:
    return MemoryRef(
        id=fields["id"], entry_id=fields["entry_id"], md_path=fields["md_path"]
    )


def _stored_episode(fields: Row) -> Stored[Episode]:
    return Stored(
        ref=_daily_ref(fields),
        scope=_scope(fields),
        item=Episode(
            owner_id=fields["owner_id"],
            image_ids=list(fields.get("image_ids") or []),
            episode=fields["episode"],
            timestamp=to_timestamp_ms(fields["timestamp"]),
            # Rows written before either column existed hold ``None``.
            subject=fields.get("subject") or "",
            summary=fields.get("summary") or "",
            session_id=fields.get("session_id"),
            sender_ids=list(fields.get("sender_ids") or []),
            parent_id=fields["parent_id"],
            parent_type=fields.get("parent_type") or "memcell",
            # Group attribution; a one-to-one row (and a row from before the
            # columns existed) holds NULL and reads the defaults.
            subject_ids=list(fields.get("subject_ids") or []),
            origin=fields.get("origin"),
            source_start=_ms_or_none(fields.get("source_start")),
        ),
    )


def _ms_or_none(value: object) -> int | None:
    """Epoch ms of a nullable timestamp column."""
    return to_timestamp_ms(value) if value is not None else None  # type: ignore[arg-type]


def _stored_image_memory(fields: Row) -> Stored[ImageMemory]:
    return Stored(
        ref=_daily_ref(fields),
        scope=_scope(fields),
        item=ImageMemory(
            owner_id=fields["owner_id"],
            description=fields["description"],
            ocr=fields.get("ocr") or "",
            timestamp=to_timestamp_ms(fields["timestamp"]),
            image_id=fields["image_id"],
            original_relpath=fields.get("original_path") or "",
            name=fields.get("name") or "",
            media_type=fields.get("media_type") or "",
            width=int(fields.get("width") or 0),
            height=int(fields.get("height") or 0),
            session_id=fields.get("session_id"),
            parent_id=fields.get("parent_id") or "",
            parent_type=fields.get("parent_type") or "memcell",
            entry_id=fields.get("entry_id"),
            md_path=fields.get("md_path"),
            status="superseded" if fields.get("deprecated_by") else "current",
            superseded_by=fields.get("deprecated_by"),
            sender_ids=list(fields.get("sender_ids") or []),
            subject_ids=list(fields.get("subject_ids") or []),
            origin=fields.get("origin"),
        ),
    )


def _stored_group_profile_items(rows: Sequence[Row]) -> list[Stored[GroupProfileItem]]:
    return [
        stored
        for stored in (_stored_group_profile_item(row) for row in rows)
        if stored is not None
    ]


def _stored_group_profile_item(fields: Row) -> Stored[GroupProfileItem] | None:
    try:
        item = GroupProfileItem.model_validate_json(fields["payload_json"])
    except ValidationError as exc:
        logger.warning(
            "group_profile_row_unreadable",
            row_id=fields.get("id"),
            md_path=fields.get("md_path"),
            error=str(exc)[:200],
        )
        return None
    item = item.read_back(item_id=fields["entry_id"], mark=fields.get("deprecated_by"))
    return Stored(ref=_daily_ref(fields), scope=_scope(fields), item=item)


def _stored_profile(fields: Row, *, subject: str) -> Stored[Profile]:
    """A profile row as the profile about ``subject``, in the row's scope."""
    return Stored(
        ref=MemoryRef(
            id=fields["id"], entry_id=fields["id"], md_path=fields["md_path"]
        ),
        scope=_scope(fields),
        item=Profile.from_stored(
            owner_id=subject,
            summary=fields["summary"],
            timestamp=fields["profile_timestamp_ms"],
            explicit_info=_json_list(fields["explicit_info_json"]),
            implicit_traits=_json_list(fields["implicit_traits_json"]),
        ),
    )


def _json_list(text: str) -> list[Any]:
    """Decode a json-encoded profile bucket; ``[]`` when empty or malformed.

    A stale encoding must not break a read; the cascade rewrites the column
    on its next reconcile.
    """
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("profile_json_decode_failed", payload_head=text[:80])
        return []
    return decoded if isinstance(decoded, list) else []
