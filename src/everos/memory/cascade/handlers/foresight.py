"""Foresight cascade handler — md → LanceDB ``foresight`` table.

Two-field BM25: ``foresight_tokens`` is the primary search column,
``evidence_tokens`` rides along when an Evidence section is present.
The vector embedding is fed only from the foresight body (evidence
is supporting context, not the retrieval anchor).

md contract:

``inline`` block:

- ``owner_id`` / ``session_id`` / ``timestamp`` — same shape as
  Episode / AtomicFact.
- ``parent_id``: source memcell id (``parent_type`` defaults to
  ``"memcell"``).
- ``sender_ids`` (optional): list rendering.
- ``start_time`` (optional): ISO-8601 window start.
- ``end_time`` (optional): ISO-8601 window end.
- ``duration_days`` (optional): integer.

``sections``:

- ``Foresight``: forward-looking inference text (embedded + BM25).
- ``Evidence`` (optional): supporting excerpt (secondary BM25 only).
"""

from __future__ import annotations

from everos.infra.persistence.lancedb import Foresight, ParentType, foresight_repo

from ._common import (
    optional_int,
    optional_iso_timestamp,
    parse_inline_list,
    require_iso_timestamp,
)
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry


class ForesightHandler(BaseDailyLogHandler):
    """Cascade handler for ``users/<u>/.foresights/foresight-*.md``."""

    kind = "foresight"
    lance_repo = foresight_repo
    content_change_keys = (
        "section:Foresight",
        "section:Evidence",
        "inline:start_time",
        "inline:end_time",
        "inline:duration_days",
    )
    """Foresight / Evidence sections + the semantic time-window inline
    fields (start_time / end_time / duration_days). Audit inline
    (owner_id / session_id / timestamp / parent_id / sender_ids) is
    excluded — changes there don't propagate."""

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
    ) -> Foresight:
        s = entry.structured
        text = s.sections.get("Foresight", "").strip()
        evidence = (s.sections.get("Evidence") or "").strip() or None
        tokens = self._deps.tokenizer.tokenize(text)
        vector = await self._deps.embedder.embed(text)
        evidence_tokens = (
            " ".join(self._deps.tokenizer.tokenize(evidence)) if evidence else None
        )
        return Foresight(
            id=f"{owner_id}_{entry.entry_id}",
            entry_id=entry.entry_id,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            session_id=s.inline.get("session_id"),
            timestamp=require_iso_timestamp(s.inline.get("timestamp")),
            start_time=optional_iso_timestamp(s.inline.get("start_time")),
            end_time=optional_iso_timestamp(s.inline.get("end_time")),
            duration_days=optional_int(s.inline.get("duration_days")),
            parent_type=s.inline.get("parent_type") or ParentType.MEMCELL.value,
            parent_id=s.inline.get("parent_id", ""),
            sender_ids=parse_inline_list(s.inline.get("sender_ids", "")),
            foresight=text,
            foresight_tokens=" ".join(tokens),
            evidence=evidence,
            evidence_tokens=evidence_tokens,
            md_path=md_path,
            content_sha256=entry.content_sha256,
            vector=vector,
        )
