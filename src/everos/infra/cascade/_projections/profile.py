"""Profile cascade handler -- a person's profile in a space -> ``user_profile``.

One handler for both files of the one profile schema (docs/group-memory.md,
6.3 and 7.4):

- ``users/<user_id>/user.md``: the row id is the user, ``owner_type = user``;
- ``groups/<group_id>/members/<member_id>.md``: ``owner_type = group``,
  ``owner_id`` the group and ``subject_id`` the member, under the row id
  :func:`~everos.infra.persistence.records.member_profile_row_id` gives --
  one no personal profile's id can equal.

Profile is a single-file kind, replaced whole on every update: no entry
markers, no per-entry diff. The row carries the typed projection of the
frontmatter; the body is display text and is not indexed.
``explicit_info`` / ``implicit_traits`` are heterogeneous LLM emissions (small
dicts, early profiles' strings), and LanceDB has no ``list[dict]`` column, so
both are json-encoded into ``explicit_info_json`` / ``implicit_traits_json``
and the recaller decodes them back at the DTO boundary. The items keep their
provenance keys there, so a group read can drop those outside an asker's
windows.

``history`` and ``pending_memcell_ids`` are not projected: the second is the
update cadence's bookkeeping, the first is read from markdown when it is
needed. What a read needs of the history goes onto the items instead: an item
that replaced others carries the values it replaced
(:func:`~everos.memory.models.with_previous_values`), so an entry a forget
takes out of the history leaves the row with the file.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from everos.infra.persistence.index import IndexRepository, primary_key
from everos.infra.persistence.markdown import MarkdownReader
from everos.infra.persistence.records import UserProfile, member_profile_row_id
from everos.memory.models import with_previous_values

from ..models import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_scope
from .base import Handler


class ProfileHandler(Handler):
    """Cascade handler for ``users/<u>/user.md`` and ``groups/<g>/members/<m>.md``."""

    kind = "user_profile"

    @property
    def index_repo(self) -> IndexRepository[Any]:
        """This kind's repository; the worker optimizes the table through it.

        Mirrors the daily-log handlers, which expose it the same way through
        :class:`BaseDailyLogHandler`.
        """
        return self._deps.index.user_profile

    content_change_keys: ClassVar[tuple[str, ...]] = (
        "frontmatter:summary",
        "frontmatter:explicit_info_json",
        "frontmatter:implicit_traits_json",
    )
    """Retrieval-relevant fields. ``profile_timestamp_ms`` is treated
    as audit (changes with every synthesis even when content is
    identical) — it lands on the row but doesn't enter the hash, so a
    timestamp-only drift skips re-upsert."""

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        parsed = await MarkdownReader.read(self._deps.memory_root.root / md_path)
        fm = parsed.frontmatter
        owner = _owner(fm, md_path)
        app_id, project_id = resolve_scope(md_path)
        summary = str(fm.get("summary", ""))
        history = fm.get("history")
        replaced = history if isinstance(history, list) else []
        explicit_info_json = _items_json(fm.get("explicit_info", []), replaced)
        implicit_traits_json = _items_json(fm.get("implicit_traits", []), replaced)
        digest = compute_content_sha256(
            {
                "frontmatter:summary": summary,
                "frontmatter:explicit_info_json": explicit_info_json,
                "frontmatter:implicit_traits_json": implicit_traits_json,
            }
        )
        prior = await self.index_repo.find_one_where(
            primary_key(app_id, project_id, owner["id"])
        )
        if prior is not None and prior.content_sha256 == digest:
            return HandlerOutcome(
                md_path=md_path, kind=self.kind, upserted=0, deleted=0, skipped=1
            )
        await self.index_repo.upsert(
            [
                UserProfile(
                    **owner,
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


def _owner(fm: dict[str, Any], md_path: str) -> dict[str, str]:
    """The row's id and owner columns, read off the frontmatter's owner keys:
    a member's file names its group and the member, ``user.md`` its user."""
    if "group_id" in fm:
        group_id = str(fm.get("group_id") or "")
        member_id = str(fm.get("member_id") or "")
        if not group_id or not member_id:
            raise ValueError(
                f"member profile md missing group_id or member_id: {md_path}"
            )
        return {
            "id": member_profile_row_id(group_id, member_id),
            "owner_id": group_id,
            "owner_type": "group",
            "subject_id": member_id,
        }
    owner_id = str(fm.get("user_id", ""))
    if not owner_id:
        raise ValueError(
            f"user_profile md missing required frontmatter user_id: {md_path}"
        )
    return {"id": owner_id, "owner_id": owner_id, "owner_type": "user"}


def _items_json(items: Any, history: list[Any]) -> str:
    """A bucket's column: its items as a read shows them, with what they replaced.

    ``sort_keys=True`` makes the digest stable across equivalent dicts;
    ``ensure_ascii=False`` keeps multibyte content readable in tooling that
    inspects the column without re-decoding.
    """
    shown = with_previous_values(items, history) if isinstance(items, list) else items
    return json.dumps(shown, sort_keys=True, ensure_ascii=False)
