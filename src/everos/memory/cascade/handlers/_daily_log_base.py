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

from everos.core.observability.logging import get_logger
from everos.core.persistence import MarkdownReader, StructuredEntry

from ..types import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_owner, resolve_scope
from .base import Handler

logger = get_logger(__name__)


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

        existing = await self.lance_repo.find_where(
            f"md_path = '{_q(md_path)}'",
            limit=10_000,
        )
        owner_id, owner_type = resolve_owner(parsed.frontmatter, md_path)
        app_id, project_id = resolve_scope(md_path)

        to_build, skipped = self._diff_entries(new_entries, existing)
        to_upsert = await self._embed_entries(
            to_build,
            owner_id,
            owner_type,
            app_id,
            project_id,
            md_path,
        )
        new_by_id = {e.entry_id for e in new_entries}
        to_delete_ids = [
            row.entry_id for row in existing if row.entry_id not in new_by_id
        ]

        await self._apply_lance_changes(to_upsert, to_delete_ids, md_path)
        await self._propagate_deprecations(
            parsed.frontmatter,
            owner_id,
            app_id,
            project_id,
        )
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=len(to_upsert),
            deleted=len(to_delete_ids),
            skipped=skipped,
        )

    @staticmethod
    def _diff_entries(
        new_entries: list[ParsedEntry],
        existing: list[Any],
    ) -> tuple[list[ParsedEntry], int]:
        """Compare new entries against existing rows, return changed + skip count."""
        existing_by_entry = {row.entry_id: row for row in existing}
        to_build: list[ParsedEntry] = []
        skipped = 0
        for entry in new_entries:
            prior = existing_by_entry.get(entry.entry_id)
            if prior is not None and prior.content_sha256 == entry.content_sha256:
                skipped += 1
                continue
            to_build.append(entry)
        return to_build, skipped

    async def _embed_entries(
        self,
        to_build: list[ParsedEntry],
        owner_id: str,
        owner_type: str,
        app_id: str,
        project_id: str,
        md_path: str,
    ) -> list[Any]:
        """Build LanceDB rows for changed entries (embed concurrently)."""
        if not to_build:
            return []
        return list(
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

    async def _apply_lance_changes(
        self,
        to_upsert: list[Any],
        to_delete_ids: list[str],
        md_path: str,
    ) -> None:
        """Flush upserts and deletes to LanceDB."""
        if to_upsert:
            await self.lance_repo.upsert(to_upsert)
        if to_delete_ids:
            in_list = ", ".join(f"'{eid}'" for eid in to_delete_ids)
            await self.lance_repo.delete(
                f"md_path = '{_q(md_path)}' AND entry_id IN ({in_list})"
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

    async def _propagate_deprecations(
        self,
        frontmatter: Any,
        owner_id: str,
        app_id: str,
        project_id: str,
    ) -> None:
        """Propagate deprecated_entries from frontmatter to LanceDB.

        The md file is the source of truth; cascade reconstructs the
        ``deprecated_by`` column on every sync/rebuild.
        """
        deprecated = getattr(frontmatter, "deprecated_entries", None)
        if not deprecated and isinstance(frontmatter, dict):
            deprecated = frontmatter.get("deprecated_entries")
        if not deprecated:
            return
        scope = (app_id, project_id)
        await asyncio.gather(
            *(
                self._mark_deprecated(owner_id, entry_id, deprecated_by_val, scope)
                for entry_id, deprecated_by_val in deprecated.items()
            )
        )

    async def _mark_deprecated(
        self,
        owner_id: str,
        entry_id: str,
        deprecated_by: str,
        scope: tuple[str, str],
    ) -> None:
        """Set ``deprecated_by`` on a LanceDB row matching ``entry_id``.

        Scoped to ``(app_id, project_id, owner_id, entry_id)`` to avoid
        cross-space collisions. A missing row is silently ignored — the
        entry may have been deleted or not yet indexed.
        """
        app_id, project_id = scope
        predicate = (
            f"owner_id = '{_q(owner_id)}' "
            f"AND entry_id = '{_q(entry_id)}' "
            f"AND app_id = '{_q(app_id)}' "
            f"AND project_id = '{_q(project_id)}'"
        )
        try:
            await self.lance_repo.update(
                {"deprecated_by": deprecated_by},
                where=predicate,
            )
        except Exception:
            logger.warning(
                "failed to mark entry deprecated",
                entry_id=entry_id,
                deprecated_by=deprecated_by,
                kind=self.kind,
                exc_info=True,
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
