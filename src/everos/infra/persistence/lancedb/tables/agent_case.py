"""LanceDB ``agent_case`` table schema.

Field set per 17_lancedb_tables_design.md §3.4. Each row records one
task an agent worked on: intent, approach, optional pivotal insight,
and a quality score. A MemCell extracted on the agent's own execution
log yields at most one AgentCase.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar

from everos.core.persistence.lancedb import BaseLanceTable, Vector

from ._parent_type import ParentType

_DIM = 1024


class AgentCase(BaseLanceTable):
    """One agent case indexed in LanceDB."""

    TABLE_NAME: ClassVar[str] = "agent_case"
    BM25_FIELDS: ClassVar[list[str]] = ["task_intent_tokens", "approach_tokens"]

    id: str
    """PK = ``<owner_id>_<entry_id>``."""

    entry_id: str
    """md-side seq id ``ac_<YYYYMMDD>_<NNNN>``."""

    owner_id: str
    """The owning ``agent_id``."""

    owner_type: str
    """Fixed ``"agent"`` for this table."""

    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``); cascade fills from md path."""

    session_id: str
    timestamp: _dt.datetime

    parent_type: str = ParentType.MEMCELL.value
    """Source pointer — always :attr:`ParentType.MEMCELL` for agent case."""

    parent_id: str
    """Source memcell id (one memcell ↔ one case)."""

    quality_score: float
    """0.0–1.0; task completion / quality estimate."""

    task_intent: str
    """≤ 50 tokens; original surface form (returned for display)."""

    task_intent_tokens: str
    """App-layer pre-tokenised ``task_intent`` — BM25 main field
    (whitespace tokenizer); display goes through ``task_intent``."""

    approach: str
    """≤ 1000 tokens; step-by-step approach (display)."""

    approach_tokens: str
    """App-layer pre-tokenised ``approach`` — BM25 secondary field
    (whitespace tokenizer). Searched in parallel with
    ``task_intent_tokens`` then merged by max score in the recall
    layer; task_intent typically scores higher because it's the
    retrieval anchor, but approach catches queries that match a step
    detail."""

    key_insight: str | None = None
    """≤ 40 tokens; pivotal strategy shift, optional."""

    md_path: str
    content_sha256: str
    """SHA-256 hex digest over the **content-bearing fields only** of
    the md entry — TaskIntent / Approach / KeyInsight sections plus
    the ``quality_score`` inline. Audit inline (owner_id /
    session_id / timestamp / parent_id) is NOT in the hash. See
    :attr:`AgentCaseHandler.content_change_keys`."""

    vector: Vector(_DIM)  # type: ignore[valid-type]
