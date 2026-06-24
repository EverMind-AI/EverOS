"""KnowledgeTopic cascade handler — md → LanceDB + SQLite.

Unlike every other handler which writes to LanceDB only, the knowledge
topic handler writes to **both** LanceDB (summary embedding + BM25
tokens) and SQLite (content full text + tree structure).  This is the
cross-storage pattern described in design spec §5.6.

Cross-storage failure is handled by cascade worker retry: if either
write fails, the handler raises, the worker marks the md_path as
failed and retries later.  Both writes are idempotent (upsert), so
retries are safe.

md contract:

- ``knowledge/{category}/{doc_title}/<n>_<name>.md`` frontmatter:
  ``type: knowledge_topic``, ``id`` / ``node_id`` / ``doc_id`` /
  ``category_id`` / ``topic_index`` / ``topic_name`` / ``topic_path``
  / ``summary`` / ``depth`` / ``parent_node_id`` / ``children_node_ids``
  / ``content_labels``.
- Body: topic content full text.

Diff strategy: SHA-256 over the **content-bearing fields** (summary,
topic_name, topic_path, category_id, depth, body).  Audit / tree
structure fields are excluded — they change on re-parse without
semantic drift.

Embedding source: ``summary`` (mirrors the search recaller's anchor).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from everos.component.utils.datetime import get_utc_now
from everos.core.persistence import MarkdownReader, ParsedMarkdown
from everos.infra.persistence.lancedb import KnowledgeTopic, knowledge_topic_repo
from everos.infra.persistence.sqlite import (
    TopicUpsertPayload,
    knowledge_topic_sqlite_repo,
)

from ..types import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_scope
from .base import Handler


class KnowledgeTopicHandler(Handler):
    """Cascade handler for
    ``knowledge/{category}/{doc_title}/<n>_<name>.md``."""

    kind = "knowledge_topic"
    lance_repo: ClassVar[Any] = knowledge_topic_repo

    content_change_keys: ClassVar[tuple[str, ...]] = (
        "frontmatter:summary",
        "frontmatter:topic_name",
        "frontmatter:topic_path",
        "frontmatter:category_id",
        "frontmatter:depth",
        "body",
    )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        absolute = self._deps.memory_root.root / md_path
        parsed = await MarkdownReader.read(absolute)

        fields = self._parse_topic_fields(parsed, md_path)
        if fields is None:
            return HandlerOutcome(
                md_path=md_path,
                kind=self.kind,
                upserted=0,
                deleted=0,
                skipped=1,
            )

        digest = compute_content_sha256(
            {
                "frontmatter:summary": fields["summary"],
                "frontmatter:topic_name": fields["topic_name"],
                "frontmatter:topic_path": fields["topic_path"],
                "frontmatter:category_id": fields["category_id"],
                "frontmatter:depth": str(fields["depth"]),
                "body": fields["content"].rstrip(),
            }
        )

        prior = await knowledge_topic_repo.get_by_id(fields["node_id"])
        if prior is not None and prior.content_sha256 == digest:
            return HandlerOutcome(
                md_path=md_path,
                kind=self.kind,
                upserted=0,
                deleted=0,
                skipped=1,
            )

        row = await self._build_lance_row(fields, digest, md_path)
        await knowledge_topic_repo.upsert([row])

        topic_payload = self._build_sqlite_payload(fields, md_path)
        await knowledge_topic_sqlite_repo.upsert_from_handler(topic_payload)

        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=1,
            deleted=0,
            skipped=0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_topic_fields(
        self,
        parsed: ParsedMarkdown,
        md_path: str,
    ) -> dict[str, Any] | None:
        """Extract frontmatter fields and body; ``None`` if type mismatch."""
        fm = parsed.frontmatter
        if fm.get("type") != "knowledge_topic":
            return None
        app_id, project_id = resolve_scope(md_path)
        return {
            "node_id": str(fm["id"]),
            "doc_id": str(fm["doc_id"]),
            "summary": str(fm.get("summary", "")),
            "content": parsed.body,
            "app_id": app_id,
            "project_id": project_id,
            "topic_name": str(fm.get("topic_name", "")),
            "topic_path": str(fm.get("topic_path", "")),
            "category_id": str(fm.get("category_id", "")),
            "depth": int(fm.get("depth", 0)),
            "parent_node_id": fm.get("parent_node_id"),
            "children_node_ids": fm.get("children_node_ids", []),
            "content_labels": fm.get("content_labels", []),
            "topic_index": int(fm.get("topic_index", 0)),
        }

    def _build_sqlite_payload(
        self,
        fields: dict[str, Any],
        md_path: str,
    ) -> TopicUpsertPayload:
        """Construct the SQLite upsert payload from parsed topic fields."""
        return TopicUpsertPayload(
            node_id=fields["node_id"],
            doc_id=fields["doc_id"],
            app_id=fields["app_id"],
            project_id=fields["project_id"],
            category_id=fields["category_id"],
            topic_index=fields["topic_index"],
            topic_name=fields["topic_name"],
            topic_path=fields["topic_path"],
            depth=fields["depth"],
            parent_node_id=fields["parent_node_id"],
            children_node_ids=(
                json.dumps(fields["children_node_ids"])
                if fields["children_node_ids"]
                else None
            ),
            summary=fields["summary"],
            content=fields["content"],
            content_labels=(
                json.dumps(fields["content_labels"])
                if fields["content_labels"]
                else None
            ),
            md_path=md_path,
        )

    async def _build_lance_row(
        self,
        fields: dict[str, Any],
        digest: str,
        md_path: str,
    ) -> KnowledgeTopic:
        """Tokenize, embed, and construct the LanceDB row."""
        summary_tokens = " ".join(
            self._deps.tokenizer.tokenize(fields["summary"]),
        )
        content_tokens = " ".join(
            self._deps.tokenizer.tokenize(fields["content"]),
        )
        vector = await self._deps.embedder.embed(fields["summary"])
        now = get_utc_now()
        return KnowledgeTopic(
            id=fields["node_id"],
            doc_id=fields["doc_id"],
            category_id=fields["category_id"],
            app_id=fields["app_id"],
            project_id=fields["project_id"],
            topic_name=fields["topic_name"],
            topic_path=fields["topic_path"],
            depth=fields["depth"],
            parent_node_id=fields["parent_node_id"] or "",
            summary=fields["summary"],
            summary_tokens=summary_tokens,
            content_tokens=content_tokens,
            content_labels=list(fields["content_labels"]),
            md_path=md_path,
            content_sha256=digest,
            vector=vector,
            created_at=now,
            updated_at=now,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        lance_deleted = await knowledge_topic_repo.delete_by_md_path(md_path)
        sqlite_deleted = await knowledge_topic_sqlite_repo.delete_by_md_path(md_path)
        deleted = max(lance_deleted, sqlite_deleted)
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=0,
            deleted=deleted,
            skipped=0,
        )
