"""Group-profile cascade handler -- md -> the ``group_profile_item`` table.

Inherits the daily-log diff / dispatch loop and maps one item entry of
``groups/<g>/profile/group_profile-<YYYY-MM-DD>.md`` into one row. The entry
format is the markdown package's (``render_group_profile_entry`` /
``group_profile_item_from_entry``, documented in
``infra/persistence/markdown/_mds/group_profile.py``); this handler reads it
through that one parser, never field by field, so the row and the store's own
read agree by construction. An entry the parser refuses raises ``ValueError``:
the worker records the file as unrecoverable instead of indexing half an item.

What the row takes from the entry:

- the link and window columns (``kind`` / ``topic_id`` / ``supersedes`` /
  ``about`` / ``member_id`` / ``role`` / the attribution lists /
  ``source_start`` / ``source_end`` / ``observed_at``);
- ``statement_tokens``: the statement, the name and the summary through the
  shared tokenizer -- the only search surface, since nothing is embedded
  (:meth:`_embed_texts` returns no text, so no embedding call is ever made);
- ``payload_json``: the whole item, which the index adapter rebuilds it from.

``deprecated_by`` comes from the file's ``deprecated_entries`` through the
base's propagation, exactly as for the other daily-log kinds: a superseding
item marks the entry it replaced, a forget marks it ``forgotten``.

Every content field is in the digest: the window and attribution fields
decide who may read an item, so an edit of any of them must reach the row.
Only the informational ``sources`` inline line is left out -- the ``Evidence``
section it summarises is in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from everos.core.utils.datetime import from_timestamp_ms
from everos.infra.persistence.index import IndexRepository
from everos.infra.persistence.markdown import group_profile_item_from_entry
from everos.infra.persistence.records import GroupProfileItem

from ._daily_log_base import BaseDailyLogHandler, ParsedEntry

_READ_SIDE = frozenset({"item_id", "status", "superseded_by"})
"""Fields a read sets on the domain item; never part of the stored payload."""


class GroupProfileHandler(BaseDailyLogHandler):
    """Cascade handler for ``groups/<g>/profile/group_profile-*.md``."""

    kind = "group_profile"

    @property
    def index_repo(self) -> IndexRepository[Any]:
        return self._deps.index.group_profile_item

    content_change_keys = (
        "section:Statement",
        "section:Name",
        "section:Summary",
        "section:Evidence",
        "inline:kind",
        "inline:topic_id",
        "inline:topic_status",
        "inline:decided_by",
        "inline:agreed_by",
        "inline:assignees",
        "inline:applies_to",
        "inline:held_by",
        "inline:about",
        "inline:resolved",
        "inline:resolved_by",
        "inline:member_id",
        "inline:role",
        "inline:confidence",
        "inline:retired",
        "inline:supersedes",
        "inline:subject_ids",
        "inline:sender_ids",
        "inline:origin",
        "inline:source_start",
        "inline:source_end",
        "inline:observed_at",
    )

    def _embed_texts(self, entry: ParsedEntry) -> tuple[str, ...]:
        return ()

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
        vectors: Sequence[list[float] | None] = (),
    ) -> GroupProfileItem:
        item = group_profile_item_from_entry(entry.structured, owner_id=owner_id)
        searchable = "\n".join(
            text for text in (item.statement, item.name, item.summary) if text
        )
        return GroupProfileItem(
            id=f"{owner_id}_{entry.entry_id}",
            entry_id=entry.entry_id,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            kind=item.kind,
            topic_id=item.topic_id,
            statement=item.statement,
            statement_tokens=" ".join(self._deps.tokenizer.tokenize(searchable)),
            supersedes=item.supersedes,
            about=item.about,
            member_id=item.member_id,
            role=item.role,
            source_memcell_ids=item.source_memcell_ids,
            subject_ids=list(item.subject_ids),
            sender_ids=list(item.sender_ids),
            decided_by=list(item.decided_by),
            agreed_by=list(item.agreed_by),
            held_by=list(item.held_by),
            origin=item.origin,
            source_start=from_timestamp_ms(item.source_start),
            source_end=from_timestamp_ms(item.source_end),
            observed_at=from_timestamp_ms(item.observed_at),
            payload_json=item.model_dump_json(exclude=set(_READ_SIDE)),
            md_path=md_path,
            content_sha256=entry.content_sha256,
        )
