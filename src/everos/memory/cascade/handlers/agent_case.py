"""AgentCase cascade handler — md → LanceDB ``agent_case`` table.

Same daily-log shape as the user-track kinds, but the path lives
under ``agents/`` and there is no ``sender_ids`` column (an agent
case has a single executing agent — the ``owner_id`` itself).

md contract:

``inline`` block:

- ``owner_id``: ``agent_id``.
- ``session_id``: conversation scope.
- ``timestamp``: ISO-8601 string.
- ``parent_id``: source memcell id (``parent_type`` defaults to
  ``"memcell"``).
- ``quality_score``: float ``0.0–1.0`` (LLM-emitted).

``sections``:

- ``TaskIntent``: short intent statement (embedded + BM25 primary).
- ``Approach``: step-by-step approach (BM25 secondary field via
  ``approach_tokens``, **not** fed to the embedder — the
  retrieval anchor is task_intent).
- ``KeyInsight`` (optional): pivotal strategy note (display only).
"""

from __future__ import annotations

from everos.infra.persistence.lancedb import AgentCase, ParentType, agent_case_repo

from ._common import require_float, require_iso_timestamp
from ._daily_log_base import BaseDailyLogHandler, ParsedEntry


class AgentCaseHandler(BaseDailyLogHandler):
    """Cascade handler for ``agents/<a>/.cases/agent_case-*.md``."""

    kind = "agent_case"
    lance_repo = agent_case_repo
    content_change_keys = (
        "section:TaskIntent",
        "section:Approach",
        "section:KeyInsight",
        "inline:quality_score",
    )
    """Includes quality_score (semantic score, not audit). Audit
    inline (owner_id / session_id / timestamp / parent_id) is excluded."""

    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
    ) -> AgentCase:
        s = entry.structured
        task_intent = s.sections.get("TaskIntent", "").strip()
        approach = s.sections.get("Approach", "").strip()
        key_insight = (s.sections.get("KeyInsight") or "").strip() or None
        intent_tokens = self._deps.tokenizer.tokenize(task_intent)
        approach_tokens = self._deps.tokenizer.tokenize(approach)
        vector = await self._deps.embedder.embed(task_intent)
        return AgentCase(
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
            quality_score=require_float(
                s.inline.get("quality_score"), field="quality_score"
            ),
            task_intent=task_intent,
            task_intent_tokens=" ".join(intent_tokens),
            approach=approach,
            approach_tokens=" ".join(approach_tokens),
            key_insight=key_insight,
            md_path=md_path,
            content_sha256=entry.content_sha256,
            vector=vector,
        )
