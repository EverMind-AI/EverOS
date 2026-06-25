"""LanceDB ``atomic_fact`` table schema.

Field set per 17_lancedb_tables_design.md §3.2. Each row carries one
atomic fact extracted by the algo layer; the parent is always the source
MemCell — recorded via ``parent_type`` / ``parent_id``.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

from ._parent_type import ParentType

_DIM = 1024


class AtomicFact(BaseLanceTable):
    """One atomic fact indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "atomic_fact"
    BM25_FIELDS: ClassVar[list[str]] = ["fact_tokens"]

    id: str
    """PK = ``<owner_id>_<entry_id>``."""

    entry_id: str
    """md-side seq id ``af_<YYYYMMDD>_<NNNN>``."""

    owner_id: str
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""
    session_id: str | None = None
    timestamp: _dt.datetime

    parent_type: str = ParentType.MEMCELL.value
    """Source pointer — always :attr:`ParentType.MEMCELL` for atomic fact."""

    parent_id: str
    """Source memcell id."""
    sender_ids: list[str]
    fact: str
    """Atomic fact text — original surface form (returned for display)."""

    fact_tokens: str
    """App-layer pre-tokenised ``fact`` text — space-joined tokens.
    BM25 index is built on this column (whitespace tokenizer);
    ``fact`` itself is what callers display."""

    md_path: str
    content_sha256: str
    """SHA-256 hex digest over the **content-bearing fields only** of
    the md entry (per :attr:`AtomicFactHandler.content_change_keys`).
    Matching digest → skip re-upsert + re-embed. Audit inline fields
    (owner_id / session_id / timestamp / parent_id / sender_ids) are
    NOT in the hash."""

    deprecated_by: str | None = None
    """Soft-delete marker set by Reflection when this fact is
    consolidated. Value is the cluster entry_id that supersedes this
    row. ``NULL`` means the row is still active."""

    vector: Vector(_DIM)  # type: ignore[valid-type]
