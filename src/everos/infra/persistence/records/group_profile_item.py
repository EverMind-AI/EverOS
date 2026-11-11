"""Derived-index ``group_profile_item`` record -- one row per group-profile item.

Rows are projected by the cascade from
``groups/<g>/profile/group_profile-<YYYY-MM-DD>.md`` (docs/group-memory.md,
sections 6.3 and 7.5), one per entry, so every version of an item is a row of
its own; ``deprecated_by`` (from the file's ``deprecated_entries``) is the
entry that replaced it, or ``forgotten``. A row's status is read from that
column, never stored: what the item states itself is in ``payload_json``.

Three kinds of column, and nothing else:

- the owner and window columns every group record has (``owner_type`` /
  ``owner_id`` / ``app_id`` / ``project_id`` / ``deprecated_by`` /
  ``source_start`` / ``source_end``), so a read scope's predicate applies to
  this table unchanged;
- the link columns a read follows, named as the domain names them (a filter
  says ``kind``, ``topic_id``, ``supersedes``, ``about``, ``member_id``,
  ``source_memcell_ids`` and is applied as is), plus ``statement_tokens`` for
  BM25 over the statement, the name and the summary;
- ``payload_json``: the whole item, which a read is rebuilt from.

No vector column: nothing about an item is embedded, so the table takes no
part in vector backfill.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from pydantic import Field

from everos.infra.persistence.records.base import IndexRecord


class GroupProfileItem(IndexRecord):
    """One version of one group-profile item in the derived index."""

    TABLE_NAME: ClassVar[str] = "group_profile_item"
    BM25_FIELDS: ClassVar[list[str]] = ["statement_tokens"]
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")

    id: str
    """``<group_id>_<entry_id>``; unique within ``(app_id, project_id)``."""

    entry_id: str
    """md-side id ``gp_<YYYYMMDD>_<seq>`` -- the domain's ``item_id``."""

    owner_id: str
    """The group."""
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"

    kind: str
    """``topic`` / ``decision`` / ``dissent`` / ``role``."""
    topic_id: str | None = None
    """Stable across a topic's versions; set on a decision filed under one."""
    statement: str = ""
    """A decision's or a dissent's statement; empty for topics and roles."""
    statement_tokens: str = ""
    """App-layer pre-tokenised statement, name and summary (BM25)."""
    supersedes: str | None = None
    """Entry id of the item this one replaced."""
    about: str | None = None
    """What a dissent is about: a decision's entry id or a topic id."""
    member_id: str | None = None
    """The member a role is observed for."""
    role: str | None = None

    source_memcell_ids: list[str] = Field(default_factory=list)
    """The segments the item's evidence came from, in evidence order."""
    subject_ids: list[str] = Field(default_factory=list)
    sender_ids: list[str] = Field(default_factory=list)
    """The evidence's speakers."""
    decided_by: list[str] = Field(default_factory=list)
    agreed_by: list[str] = Field(default_factory=list)
    held_by: list[str] = Field(default_factory=list)
    origin: str | None = None
    """``member`` / ``assistant`` / ``mixed``: whose messages the evidence is."""

    source_start: _dt.datetime
    """The earliest evidence message; a group read compares it with the
    asker's window start (``source_start >= from``)."""
    source_end: _dt.datetime
    """The latest evidence message (``source_end < to``)."""
    observed_at: _dt.datetime
    """When the update that wrote the item ran (its batch's last segment);
    newest first is the order of every listing."""

    payload_json: str
    """The domain item as JSON, without its read-side fields (``item_id`` /
    ``status`` / ``superseded_by``)."""

    md_path: str
    content_sha256: str
    """SHA-256 over every content field of the md entry; a matching digest on
    reconcile skips the row."""

    deprecated_by: str | None = None
    """The entry that replaced this one, ``forgotten`` after a forget, ``NULL``
    while it is the newest version."""
