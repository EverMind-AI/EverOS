"""LanceDB ``foresight`` table schema.

Field set per 17_lancedb_tables_design.md §3.3. Each row carries a
forward-looking inference about the user (intent window, planned
action, projected need); ``start_time`` / ``end_time`` describe the
window the foresight applies to.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

from ._parent_type import ParentType

_DIM = 1024


class Foresight(BaseLanceTable):
    """One foresight record indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "foresight"
    BM25_FIELDS: ClassVar[list[str]] = ["foresight_tokens", "evidence_tokens"]

    id: str
    """PK = ``<owner_id>_<entry_id>``."""

    entry_id: str
    """md-side seq id ``fs_<YYYYMMDD>_<NNNN>``."""

    owner_id: str
    owner_type: str
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""
    session_id: str | None = None
    timestamp: _dt.datetime
    """Foresight generation time."""

    start_time: _dt.datetime | None = None
    """Foresight coverage window start; tz-aware."""

    end_time: _dt.datetime | None = None
    """Foresight coverage window end; tz-aware."""

    duration_days: int | None = None

    parent_type: str = ParentType.MEMCELL.value
    """Source pointer — always :attr:`ParentType.MEMCELL` for foresight."""

    parent_id: str
    """Source memcell id."""

    sender_ids: list[str]
    foresight: str
    """Foresight body — original surface form (returned for display)."""

    foresight_tokens: str
    """App-layer pre-tokenised ``foresight`` text — space-joined tokens.
    BM25 index is built on this column (whitespace tokenizer)."""

    evidence: str | None = None
    """Supporting evidence excerpt; may be empty."""

    evidence_tokens: str | None = None
    """App-layer pre-tokenised ``evidence`` (secondary BM25 field).
    ``None`` whenever ``evidence`` is None."""

    md_path: str
    content_sha256: str
    """SHA-256 hex digest over the **content-bearing fields only** of
    the md entry — Foresight / Evidence sections plus the time-window
    inline fields (start_time / end_time / duration_days). Audit inline
    (owner_id / session_id / timestamp / parent_id / sender_ids) is NOT
    in the hash. See :attr:`ForesightHandler.content_change_keys`."""

    vector: Vector(_DIM)  # type: ignore[valid-type]
