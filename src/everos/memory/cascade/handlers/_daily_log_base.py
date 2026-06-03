"""Shared diff / dispatch loop for every daily-log cascade handler.

The 4 daily-log kinds (episode / atomic_fact / foresight / agent_case)
all do the same three-way reconcile against LanceDB:

1. Parse the md into structured entries.
2. Fetch existing rows for the same ``md_path``.
3. ``content_sha256`` mismatch → tokenise + embed + upsert; no diff
   → skip; row gone from md → delete.

The hash covers **only content-bearing fields** declared by each
subclass in :attr:`content_change_keys` (a tuple of ``"section:Name"``
/ ``"inline:name"`` strings). Audit inline fields (owner_id /
session_id / timestamp / parent_id / sender_ids) are NOT in the hash
— editing them does NOT propagate to LanceDB and does NOT waste an
embed call.

Subclasses bind their ``kind`` / ``lance_repo`` / ``content_change_keys``
as ClassVars and override :meth:`_build_row` to do the per-kind field
mapping. Everything else — read, diff, embed call, upsert, delete —
lives here.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
from typing import Any, ClassVar

from everos.core.persistence import MarkdownReader, StructuredEntry

from ..types import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_owner, resolve_scope
from .base import Handler


@dataclasses.dataclass(frozen=True)
class ParsedEntry:
    """One md-side entry, parsed and digested for diff.

    Held immutable so the diff loop can hash / compare freely.
    """

    entry_id: str
    structured: StructuredEntry
    content_sha256: str


class BaseDailyLogHandler(Handler):
    """Common chassis for the 4 daily-log cascade handlers.

    Subclass requirements:

    - :attr:`kind` (ClassVar[str]) — registry name, surfaces in logs.
    - :attr:`lance_repo` (ClassVar) — the LanceDB repo singleton for
      this kind (must expose ``find_where`` / ``upsert`` / ``delete``
      / ``delete_by_md_path``).
    - :attr:`content_change_keys` (ClassVar[tuple[str, ...]]) — the
      subset of inline + section fields whose changes should trigger
      re-upsert + re-embed. Each key is ``"section:Name"`` or
      ``"inline:name"``.
    - :meth:`_build_row` (override) — turn a :class:`ParsedEntry` plus
      common context (owner_id / owner_type / md_path) into a typed
      LanceDB row. Tokenisation + embedding live in the subclass.
    """

    kind: ClassVar[str] = ""
    lance_repo: ClassVar[Any] = None
    content_change_keys: ClassVar[tuple[str, ...]] = ()

    def _content_sha256(self, structured: StructuredEntry) -> str:
        """Hash the content-bearing subset of one entry's inline+sections.

        Walks :attr:`content_change_keys`, projects each key onto its
        ``section:`` / ``inline:`` source on the structured entry, and
        canonicalises into a digest. Unknown key prefixes raise
        :class:`ValueError` so a typo on a subclass surfaces immediately.
        """
        parts: dict[str, str] = {}
        for key in self.content_change_keys:
            kind, _, name = key.partition(":")
            if kind == "section":
                parts[key] = structured.sections.get(name) or ""
            elif kind == "inline":
                parts[key] = structured.inline.get(name) or ""
            else:
                raise ValueError(
                    f"{type(self).__name__}.content_change_keys has unsupported "
                    f"prefix in {key!r}; expected 'section:' or 'inline:'"
                )
        return compute_content_sha256(parts)

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        absolute = self._deps.memory_root.root / md_path
        parsed = await MarkdownReader.read(absolute)
        new_entries = [
            ParsedEntry(
                entry_id=entry.id,
                structured=entry.as_structured(),
                content_sha256=self._content_sha256(entry.as_structured()),
            )
            for entry in parsed.entries
        ]
        new_by_id = {e.entry_id: e for e in new_entries}

        existing = await self.lance_repo.find_where(
            f"md_path = '{_q(md_path)}'",
            limit=10_000,
        )
        existing_by_entry = {row.entry_id: row for row in existing}

        owner_id, owner_type = resolve_owner(parsed.frontmatter, md_path)
        app_id, project_id = resolve_scope(md_path)

        to_build: list[ParsedEntry] = []
        skipped = 0
        for entry in new_entries:
            prior = existing_by_entry.get(entry.entry_id)
            if prior is not None and prior.content_sha256 == entry.content_sha256:
                skipped += 1
                continue
            to_build.append(entry)

        # Build rows concurrently; ``_build_row`` calls ``embedder.embed``
        # which is already capped by a process-global ``asyncio.Semaphore``
        # at ``max_concurrent`` (see OpenAIEmbeddingProvider). This unblocks
        # per-md-path embedding pipelining without uncapping embed-API rate.
        to_upsert: list[Any] = (
            list(
                await asyncio.gather(
                    *(
                        self._build_row(
                            owner_id=owner_id,
                            owner_type=owner_type,
                            app_id=app_id,
                            project_id=project_id,
                            md_path=md_path,
                            entry=entry,
                        )
                        for entry in to_build
                    )
                )
            )
            if to_build
            else []
        )

        to_delete_ids = [
            row.entry_id for row in existing if row.entry_id not in new_by_id
        ]

        if to_upsert:
            await self.lance_repo.upsert(to_upsert)
        if to_delete_ids:
            in_list = ", ".join(f"'{eid}'" for eid in to_delete_ids)
            await self.lance_repo.delete(
                f"md_path = '{_q(md_path)}' AND entry_id IN ({in_list})"
            )

        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=len(to_upsert),
            deleted=len(to_delete_ids),
            skipped=skipped,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await self.lance_repo.delete_by_md_path(md_path)
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=0,
            deleted=deleted,
            skipped=0,
        )

    @abc.abstractmethod
    async def _build_row(
        self,
        *,
        owner_id: str,
        owner_type: str,
        app_id: str = "default",
        project_id: str = "default",
        md_path: str,
        entry: ParsedEntry,
    ) -> Any:
        """Subclass: build the typed LanceDB row for one parsed entry.

        ``app_id`` / ``project_id`` carry the path-derived scope; the base
        always supplies them (via :func:`resolve_scope`). They default to
        ``"default"`` so white-box callers exercising only the field mapping
        can omit them.
        """


def _q(text: str) -> str:
    """Defensive SQL-quote escape (mirrors lancedb chassis convention)."""
    return text.replace("'", "''")
