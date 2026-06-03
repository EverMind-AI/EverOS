"""LanceDB ``episode`` table schema.

Field set is fixed by the LanceDB tables design spec. Rows are populated
by the cascade daemon from ``users/<owner_id>/episodes/episode-<YYYY-MM-DD>.md``
and from ``agents/<owner_id>/episodes/...`` symmetrically.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

from ._parent_type import ParentType

# Vector dimension is settings-managed at runtime; the class-level
# constant pins the schema dim used at table creation.
_DIM = 1024


class Episode(BaseLanceTable):
    """One episode record indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "episode"
    BM25_FIELDS: ClassVar[list[str]] = ["episode_tokens"]

    id: str
    """PK = ``<owner_id>_<entry_id>`` (scalar PK)."""

    entry_id: str
    """md-side seq id ``ep_<YYYYMMDD>_<NNNN>`` (cascade reverse-lookup)."""

    owner_id: str
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""
    session_id: str
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

    subject: str | None = None
    summary: str | None = None
    episode: str
    """Full narrative text — original surface form (returned for display)."""

    episode_tokens: str
    """App-layer pre-tokenised ``episode`` text — space-joined tokens
    (e.g. produced by jieba). LanceDB FTS index is built on **this**
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
    NOT in the hash so editing them doesn't waste an embedding call.
    See ``16_cascade_impl_design.md`` §3.3."""

    vector: Vector(_DIM)  # type: ignore[valid-type]
