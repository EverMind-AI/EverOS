"""Principle cascade handler — md → LanceDB ``principle`` table.

``principles.md`` is a single-file kind (same chassis as
``users/<user_id>/user.md``): one file per user, replaced wholesale.
Unlike :class:`UserProfileHandler` (one row per file), this handler
explodes the frontmatter ``principles`` list into N Lance rows
(``id = <owner_id>_<principle_id>``).

md contract:

- frontmatter: :class:`PrincipleFrontmatter` (``user_id`` + structured
  ``principles`` list).
- body: human-readable markdown list (not indexed).

``timestamp_ms`` is audit: it lands on the row but is excluded from
``content_sha256``, so a timestamp-only drift skips re-upsert.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from everos.core.persistence import MarkdownReader
from everos.infra.persistence.lancedb import Principle, principle_repo

from ..types import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_scope
from .base import Handler


class PrincipleHandler(Handler):
    """Cascade handler for ``users/<user_id>/principles.md``."""

    kind = "principle"
    lance_repo: ClassVar[Any] = principle_repo
    """Exposed for ``CascadeWorker._optimize_touched_kinds``."""

    content_change_keys: ClassVar[tuple[str, ...]] = (
        "frontmatter:title",
        "frontmatter:statement",
        "frontmatter:source_entry_ids_json",
    )

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        absolute = self._deps.memory_root.root / md_path
        parsed = await MarkdownReader.read(absolute)
        fm = parsed.frontmatter

        owner_id = str(fm.get("user_id", ""))
        if not owner_id:
            raise ValueError(
                f"principle md missing required frontmatter user_id: {md_path}"
            )
        app_id, project_id = resolve_scope(md_path)

        items = _parse_items(fm.get("principles", []))
        seen_ids: set[str] = set()
        desired: dict[str, Principle] = {}
        for item in items:
            principle_id = str(item.get("id") or "").strip()
            if not principle_id:
                raise ValueError(f"principle md has an item with empty id: {md_path}")
            if principle_id in seen_ids:
                raise ValueError(
                    f"principle md has duplicate id {principle_id!r}: {md_path}"
                )
            seen_ids.add(principle_id)
            title = str(item.get("title", ""))
            statement = str(item.get("statement", ""))
            source_entry_ids = _as_str_list(item.get("source_entry_ids", []))
            source_json = json.dumps(
                source_entry_ids, sort_keys=True, ensure_ascii=False
            )
            digest = compute_content_sha256(
                {
                    "frontmatter:title": title,
                    "frontmatter:statement": statement,
                    "frontmatter:source_entry_ids_json": source_json,
                }
            )
            row_id = f"{owner_id}_{principle_id}"
            desired[row_id] = Principle(
                id=row_id,
                principle_id=principle_id,
                owner_id=owner_id,
                owner_type="user",
                app_id=app_id,
                project_id=project_id,
                title=title,
                statement=statement,
                source_entry_ids=source_entry_ids,
                timestamp_ms=int(item.get("timestamp_ms") or 0),
                md_path=md_path,
                content_sha256=digest,
            )

        existing = await principle_repo.find_where(
            f"md_path = '{_q(md_path)}'",
            limit=10_000,
        )
        existing_by_id = {row.id: row for row in existing}

        to_upsert = [
            row
            for row_id, row in desired.items()
            if existing_by_id.get(row_id) is None
            or existing_by_id[row_id].content_sha256 != row.content_sha256
        ]
        to_delete_ids = [row.id for row in existing if row.id not in desired]
        skipped = len(desired) - len(to_upsert)

        if to_upsert:
            await principle_repo.upsert(to_upsert)
        if to_delete_ids:
            in_list = ", ".join(f"'{_q(rid)}'" for rid in to_delete_ids)
            await principle_repo.delete(
                f"md_path = '{_q(md_path)}' AND id IN ({in_list})"
            )

        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=len(to_upsert),
            deleted=len(to_delete_ids),
            skipped=skipped,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await principle_repo.delete_by_md_path(md_path)
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=0,
            deleted=deleted,
            skipped=0,
        )


def _parse_items(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("principle frontmatter principles must be a list")
    items: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("principle frontmatter item must be a mapping")
        items.append(item)
    return items


def _as_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    return [str(v) for v in raw]


def _q(text: str) -> str:
    """Defensive SQL-quote escape (mirrors daily-log handler convention)."""
    return text.replace("'", "''")
