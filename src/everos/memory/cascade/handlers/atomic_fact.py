"""AtomicFact cascade handler — md → LanceDB ``atomic_fact`` table.

md contract (md writer + cascade share this shape):

``inline`` block:

- ``owner_id``: ``user_id`` (atomic facts are user-track today).
- ``session_id``: conversation scope.
- ``timestamp``: ISO-8601 string.
- ``parent_id``: source memcell id (``parent_type`` is always
  ``"memcell"`` — emitted via the schema default).
- ``sender_ids`` (optional): ``[u_a, u_b]`` rendering.

``sections``:

- ``Fact``: the atomic-fact sentence — fed to the embedder and the
  tokenizer (``fact_tokens`` BM25 field).
"""

from __future__ import annotations

from everos.infra.persistence.lancedb import AtomicFact, ParentType, atomic_fact_repo

from ._common import parse_inline_list, require_iso_timestamp
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry


class AtomicFactHandler(BaseDailyLogHandler):
    """Cascade handler for ``users/<u>/.atomic_facts/atomic_fact-*.md``."""

    kind = "atomic_fact"
    lance_repo = atomic_fact_repo
    content_change_keys = ("section:Fact",)
    """Only ``Fact`` matters — it's both the embedded text AND the
    BM25 source. Audit inline is excluded."""

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
    ) -> AtomicFact:
        s = entry.structured
        text = s.sections.get("Fact", "").strip()
        tokens = self._deps.tokenizer.tokenize(text)
        vector = await self._deps.embedder.embed(text)
        return AtomicFact(
            id=f"{owner_id}_{entry.entry_id}",
            entry_id=entry.entry_id,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            session_id=s.inline.get("session_id", ""),
            timestamp=require_iso_timestamp(s.inline.get("timestamp")),
            parent_type=s.inline.get("parent_type") or ParentType.MEMCELL.value,
            parent_id=s.inline.get("parent_id", ""),
            sender_ids=parse_inline_list(s.inline.get("sender_ids", "")),
            fact=text,
            fact_tokens=" ".join(tokens),
            md_path=md_path,
            content_sha256=entry.content_sha256,
            vector=vector,
        )
