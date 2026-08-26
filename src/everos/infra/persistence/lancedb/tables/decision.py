"""LanceDB ``decision`` table schema.

Field set for the decision LanceDB row. Each row records one committed
trade-off extracted from a MemCell (title, decision body, reason,
optional impact, tags). ``deprecated_by`` is present from day one so
Decision reflection can soft-deprecate a superseded entry.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

from ._parent_type import ParentType

_DIM = 1024


class Decision(BaseLanceTable):
    """One decision record indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "decision"
    BM25_FIELDS: ClassVar[list[str]] = ["decision_tokens", "reason_tokens"]

    id: str
    """PK = ``<owner_id>_<entry_id>``."""

    entry_id: str
    """md-side seq id ``dc_<YYYYMMDD>_<NNNN>``."""

    owner_id: str
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""
    session_id: str | None = None
    timestamp: _dt.datetime

    parent_type: str = ParentType.MEMCELL.value
    """Source pointer — always :attr:`ParentType.MEMCELL` for decision."""

    parent_id: str
    """Source memcell id."""

    title: str
    decision: str
    """Decision body — original surface form (returned for display).
    Cascade embed and backfill Phase 1 both read this column."""

    reason: str
    """Why the trade-off was made — original surface form."""

    impact: str | None = None
    tags: list[str]
    """Caller-supplied labels (not conversation ``sender_ids``)."""

    decision_tokens: str
    """App-layer pre-tokenised ``decision`` text — space-joined tokens.
    Primary BM25 column (whitespace tokenizer)."""

    reason_tokens: str
    """App-layer pre-tokenised ``reason`` — secondary BM25 column.
    Required whenever ``reason`` is (domain ``reason`` is required)."""

    md_path: str
    content_sha256: str
    """SHA-256 hex digest over the **content-bearing fields only** of
    the md entry. Audit inline (owner_id / session_id / timestamp /
    parent_id) is NOT in the hash. The exact key set is owned by the
    cascade handler's ``content_change_keys`` (lands with the handler)."""

    vector: Vector(_DIM) | None = None  # type: ignore[valid-type]
    deprecated_by: str | None = None
    """Soft-delete marker set by Decision reflection. Value is the
    superseding entry_id. ``NULL`` means the row is still active."""
