"""Member-in-group profile cascade handler -- ``members/<m>.md`` -> ``user_profile``.

docs/group-memory.md, section 6.3: a member's profile inside a group is a row
of the ``user_profile`` table with ``owner_type = group``, ``owner_id`` the
group and ``subject_id`` the member, under the row id
:func:`~everos.infra.persistence.records.member_profile_row_id` gives -- one no
personal profile's id can equal. Like the personal profile it is a single-file
kind replaced whole on every update; the items keep their provenance keys in
the JSON columns, so a read can drop the ones outside an asker's windows.

``pending_memcell_ids`` and ``history`` are not projected: the first is the
update throttle's bookkeeping, the second is read from markdown when it is
needed.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from everos.infra.persistence.index import IndexRepository, primary_key
from everos.infra.persistence.markdown import MarkdownReader
from everos.infra.persistence.records import UserProfile, member_profile_row_id

from ..models import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_scope
from .base import Handler


class GroupMemberProfileHandler(Handler):
    """Cascade handler for ``groups/<g>/members/<member_id>.md``."""

    kind = "group_member_profile"

    @property
    def index_repo(self) -> IndexRepository[Any]:
        return self._deps.index.user_profile

    content_change_keys: ClassVar[tuple[str, ...]] = (
        "frontmatter:summary",
        "frontmatter:explicit_info_json",
        "frontmatter:implicit_traits_json",
    )
    """The personal profile's digest keys: what a read returns."""

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        parsed = await MarkdownReader.read(self._deps.memory_root.root / md_path)
        fm = parsed.frontmatter
        group_id = str(fm.get("group_id", ""))
        member_id = str(fm.get("member_id", ""))
        if not group_id or not member_id:
            raise ValueError(
                f"member profile md missing group_id or member_id: {md_path}"
            )
        app_id, project_id = resolve_scope(md_path)
        summary = str(fm.get("summary", ""))
        explicit_info_json = _dump_json(fm.get("explicit_info", []))
        implicit_traits_json = _dump_json(fm.get("implicit_traits", []))
        digest = compute_content_sha256(
            {
                "frontmatter:summary": summary,
                "frontmatter:explicit_info_json": explicit_info_json,
                "frontmatter:implicit_traits_json": implicit_traits_json,
            }
        )
        row_id = member_profile_row_id(group_id, member_id)
        prior = await self.index_repo.find_one_where(
            primary_key(app_id, project_id, row_id)
        )
        if prior is not None and prior.content_sha256 == digest:
            return HandlerOutcome(
                md_path=md_path, kind=self.kind, upserted=0, deleted=0, skipped=1
            )
        await self.index_repo.upsert(
            [
                UserProfile(
                    id=row_id,
                    owner_id=group_id,
                    owner_type="group",
                    subject_id=member_id,
                    app_id=app_id,
                    project_id=project_id,
                    summary=summary,
                    explicit_info_json=explicit_info_json,
                    implicit_traits_json=implicit_traits_json,
                    profile_timestamp_ms=int(fm.get("profile_timestamp_ms", 0)),
                    md_path=md_path,
                    content_sha256=digest,
                )
            ]
        )
        return HandlerOutcome(
            md_path=md_path, kind=self.kind, upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await self.index_repo.delete_by_md_path(md_path)
        return HandlerOutcome(
            md_path=md_path, kind=self.kind, upserted=0, deleted=deleted, skipped=0
        )


def _dump_json(value: Any) -> str:
    """The personal profile handler's canonical encoding of an item bucket."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
