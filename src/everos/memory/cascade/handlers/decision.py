"""Decision cascade handler — md → LanceDB ``decision`` table.

Two-field BM25: ``decision_tokens`` is the primary search column,
``reason_tokens`` rides along from the Reason section. The vector
embedding is fed only from the Decision body (reason / title / impact
are supporting context, not the retrieval anchor).

md contract (must match ``extract_decision._decision_to_entry_body``):

``inline`` block:

- ``owner_id`` / ``session_id`` / ``timestamp`` — same shape as
  Episode / Foresight.
- ``parent_id``: source memcell id (``parent_type`` defaults to
  ``"memcell"``).
- ``tags`` (optional): list rendering ``[runtime, rust]``. Not
  conversation ``sender_ids``.

``sections``:

- ``Title``: short label (Lance ``title`` column; hashed so edits
  propagate, not embedded).
- ``Decision``: committed trade-off text (embedded + BM25 primary).
  Embedding is a soft dependency: when unavailable, ``vector`` is
  written as ``None`` and the row stays BM25/scalar-searchable only.
- ``Reason``: why the trade-off was made (secondary BM25 only).
- ``Impact`` (optional): consequence note (display only).
"""

from __future__ import annotations

from everos.component.embedding import get_embedding_capability
from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb import Decision, ParentType, decision_repo

from ._common import parse_inline_list, require_iso_timestamp
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry

logger = get_logger(__name__)


class DecisionHandler(BaseDailyLogHandler):
    """Cascade handler for ``users/<u>/decisions/decision-*.md``."""

    kind = "decision"
    lance_repo = decision_repo
    content_change_keys = (
        "section:Title",
        "section:Decision",
        "section:Reason",
        "section:Impact",
        "inline:tags",
    )
    """Title / Decision / Reason / Impact + tags. Audit inline
    (owner_id / session_id / timestamp / parent_id) is excluded —
    changes there don't propagate. Title is hashed so the Lance
    ``title`` column stays in sync even though it is not embedded."""

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
    ) -> Decision:
        s = entry.structured
        title = s.sections.get("Title", "").strip()
        text = s.sections.get("Decision", "").strip()
        reason = s.sections.get("Reason", "").strip()
        impact = (s.sections.get("Impact") or "").strip() or None
        decision_tokens = self._deps.tokenizer.tokenize(text)
        reason_tokens = self._deps.tokenizer.tokenize(reason)
        vector = await get_embedding_capability().embed_or_none(text)
        if vector is None:
            logger.debug(
                "cascade_handler_embed_skipped",
                kind=self.kind,
                entry_id=entry.entry_id,
                reason="embedding_capability_unavailable",
            )
        return Decision(
            id=f"{owner_id}_{entry.entry_id}",
            entry_id=entry.entry_id,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            session_id=s.inline.get("session_id"),
            timestamp=require_iso_timestamp(s.inline.get("timestamp")),
            parent_type=s.inline.get("parent_type") or ParentType.MEMCELL.value,
            parent_id=s.inline.get("parent_id", ""),
            title=title,
            decision=text,
            reason=reason,
            impact=impact,
            tags=parse_inline_list(s.inline.get("tags") or ""),
            decision_tokens=" ".join(decision_tokens),
            reason_tokens=" ".join(reason_tokens),
            md_path=md_path,
            content_sha256=entry.content_sha256,
            vector=vector,
        )
