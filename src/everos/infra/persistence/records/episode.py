"""Derived-index ``episode`` record.

Field set is fixed by the index tables design spec. Rows are populated
by the cascade daemon from ``users/<owner_id>/episodes/episode-<YYYY-MM-DD>.md``
and from ``agents/<owner_id>/episodes/...`` symmetrically.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from pydantic import Field

from everos.infra.persistence.records.base import IndexRecord

from ._parent_type import ParentType
from .vector import DenseVector


class Episode(IndexRecord):
    """One episode record in the derived index."""

    TABLE_NAME: ClassVar[str] = "episode"
    BM25_FIELDS: ClassVar[list[str]] = ["episode_tokens"]
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")

    id: str
    """``<owner_id>_<entry_id>``; unique within ``(app_id, project_id)`` — the
    row key is the declared ``PRIMARY_KEY``."""

    entry_id: str
    """md-side seq id ``ep_<YYYYMMDD>_<NNNN>`` (cascade reverse-lookup)."""

    owner_id: str
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""
    session_id: str | None = None
    timestamp: _dt.datetime

    parent_type: str = ParentType.MEMCELL.value
    """Source pointer — always :attr:`ParentType.MEMCELL` for episode."""

    parent_id: str
    """Source memcell id. The pipeline knows the memcell currently being
    processed and writes its id into the md entry's inline block; the
    cascade handler reads it back. The new everalgo Episode type no
    longer emits ``parent_id`` itself (collapsed to caller-managed),
    so this is filled entirely from everos's engineering context."""

    sender_ids: list[str]
    """Distinct ``role=user|assistant`` senders behind the episode."""

    image_ids: list[str] = Field(default_factory=list)
    """Content hashes of the pictures the narrative refers to, in arrival order.

    What makes the body's ``(img-1: <hash>)`` resolvable: a reader can ask the
    picture table for these ids and get the description and in-image text the
    episode deliberately does not repeat. Empty for an episode that carried no
    picture, and for every row written before the field existed.
    """

    subject: str | None = None
    summary: str | None = None
    episode: str
    """Full narrative text — original surface form (returned for display)."""

    episode_tokens: str
    """App-layer pre-tokenised ``episode`` text — space-joined tokens
    (e.g. produced by jieba). The BM25 index is built on **this**
    column using a whitespace tokenizer; the original ``episode`` field
    is what callers display. Two-field BM25 scheme keeps tokenisation
    deterministic and provider-pluggable at the app layer."""

    md_path: str
    content_sha256: str
    """SHA-256 hex digest over the **content-bearing fields only** of the
    md entry (per :attr:`EpisodeHandler.content_change_keys`). On
    re-reconcile, a matching digest means none of the persistence /
    embedding-relevant fields changed — the entry is skipped (no
    re-upsert, no re-embed). Inline audit fields (owner_id /
    session_id / timestamp / parent_id / sender_ids) are intentionally
    NOT in the hash so editing them doesn't waste an embedding call."""

    deprecated_by: str | None = None
    """Soft-delete marker set by Reflection when this episode is
    consolidated into a cluster. Value is the cluster entry_id that
    supersedes this row. ``NULL`` means the row is still active."""

    # Group attribution and the membership-window columns
    # (docs/group-memory.md, section 6.3). Filled for ``owner_type = group``
    # rows; a one-to-one row leaves them empty and reads ``source_start`` /
    # ``source_end`` as ``timestamp``.
    subject_ids: list[str] = Field(default_factory=list)
    """Members the episode is about."""
    origin: str | None = None
    """``member`` / ``assistant`` / ``mixed``."""
    source_start: _dt.datetime | None = None
    """The segment's first message. A group read compares it with the asker's
    window start: ``source_start >= from``."""
    source_end: _dt.datetime | None = None
    """The segment's last message (the episode's ``timestamp``); compared with
    the window end: ``source_end < to``."""

    vector: DenseVector | None = None
    subject_vector: DenseVector | None = None
