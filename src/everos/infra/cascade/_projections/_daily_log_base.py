"""Shared diff / dispatch loop for every daily-log cascade handler.

The daily-log kinds (episode / image_memory / group_profile) all do the same
three-way reconcile against the derived index:

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

Subclasses bind their ``kind`` / ``content_change_keys`` as ClassVars,
expose their table's repository as ``index_repo``, and override
:meth:`_build_row` to do the per-kind field mapping. Everything else —
read, diff, embed call, upsert, delete — lives here.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
from collections.abc import Collection, Sequence
from typing import Any, ClassVar

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import IndexRepository, all_of, eq, one_of
from everos.infra.persistence.markdown import MarkdownReader, StructuredEntry

from ..models import HandlerOutcome
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
    """Common chassis for the daily-log cascade handlers.

    Subclass requirements:

    - :attr:`kind` (ClassVar[str]) — registry name, surfaces in logs.
    - :attr:`index_repo` (property) — this kind's repository, taken from
      ``HandlerDeps.index``.
    - :attr:`content_change_keys` (ClassVar[tuple[str, ...]]) — the
      subset of inline + section fields whose changes should trigger
      re-upsert + re-embed. Each key is ``"section:Name"`` or
      ``"inline:name"``.
    - :meth:`_embed_texts` (override) — the texts one entry needs embedded.
    - :meth:`_build_row` (override) — turn a :class:`ParsedEntry` plus
      common context (owner_id / owner_type / md_path) and the vectors
      for its texts into a typed LanceDB row. Tokenisation lives in the
      subclass; embedding is batched here, one call per file.
    """

    kind: ClassVar[str] = ""
    content_change_keys: ClassVar[tuple[str, ...]] = ()

    @property
    @abc.abstractmethod
    def index_repo(self) -> IndexRepository[Any]:
        """This kind's derived-index repository."""

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
        # Registered before the file is read, ended however the run ends:
        # a deletion that fences rows while this run is going keeps them out
        # of this run's upsert (``_apply_lance_changes``).
        fence = self._deps.fence
        token = fence.register_run() if fence is not None else None
        try:
            return await self._project(md_path)
        finally:
            if fence is not None and token is not None:
                fence.end_run(token)

    async def _project(self, md_path: str) -> HandlerOutcome:
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

        existing = await self.index_repo.find_where(
            eq("md_path", md_path),
            limit=10_000,
            skip_vectors=True,
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
        # Rows a memory deletion fenced while this run was going stay out.
        fence = self._deps.fence
        if fence is not None:
            to_upsert = [row for row in to_upsert if not fence.blocks(row)]
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
            entry_ids=new_by_id,
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
        """Build LanceDB rows for changed entries.

        Every text the changed entries need goes out in ONE ``embed_batch``
        call (the provider chunks it by ``batch_size``), not one ``embed``
        call per entry: the per-entry fan-out queued N single-input requests
        on the process-wide embedding semaphore ahead of the interactive
        query embedding (issue #486).

        A text whose content is empty (or whitespace only) is **not** sent:
        providers reject an empty string inside the input array (TEI answers
        ``400 Input validation error: `inputs` cannot be empty``), and the
        batch is all-or-nothing, so one blank section would fail the file's
        every other entry with it. Such a text is embedded as ``None``, the
        same soft-degrade a missing provider produces, so the row is written
        without a vector and ``vector_backfill`` re-embeds it once the md
        carries content.
        """
        if not to_build:
            return []
        texts_per_entry = [self._embed_texts(entry) for entry in to_build]
        flat = [text for texts in texts_per_entry for text in texts]
        vectors: Sequence[list[float] | None]
        embedding = self._deps.embedding
        if embedding is None:
            logger.debug(
                "cascade_handler_embed_skipped",
                kind=self.kind,
                md_path=md_path,
                entries=len(to_build),
                reason="embedding_capability_unavailable",
            )
            vectors = [None] * len(flat)
        else:
            # Positions that carry content, and a placeholder result aligned
            # with ``flat`` so the per-entry slicing below is unchanged.
            positions = [i for i, text in enumerate(flat) if text.strip()]
            resolved: list[list[float] | None] = [None] * len(flat)
            if positions:
                embedded = await embedding.embed_batch([flat[i] for i in positions])
                for i, vec in zip(positions, embedded, strict=True):
                    resolved[i] = list(vec)
            vectors = resolved
            blank = len(flat) - len(positions)
            if blank:
                logger.debug(
                    "cascade_handler_embed_blank_text",
                    kind=self.kind,
                    md_path=md_path,
                    entries=len(to_build),
                    blank=blank,
                )
        rows: list[Any] = []
        cursor = 0
        for entry, texts in zip(to_build, texts_per_entry, strict=True):
            entry_vectors = vectors[cursor : cursor + len(texts)]
            cursor += len(texts)
            rows.append(
                await self._build_row(
                    owner_id=owner_id,
                    owner_type=owner_type,
                    app_id=app_id,
                    project_id=project_id,
                    md_path=md_path,
                    entry=entry,
                    vectors=entry_vectors,
                )
            )
        return await self._attach_side_vectors(rows)

    async def _attach_side_vectors(self, rows: list[Any]) -> list[Any]:
        """Fill any embedding the text batch above could not produce.

        The text batch embeds strings; a picture's own embedding has to be
        produced from the *file*, which only the picture handler knows how to
        locate. This is the seam: the base writes the text vector, the
        subclass adds whatever else its row carries. Default is a no-op, so
        every text-only kind keeps the exact behaviour it had.
        """
        return rows

    async def _apply_lance_changes(
        self,
        to_upsert: list[Any],
        to_delete_ids: list[str],
        md_path: str,
    ) -> None:
        """Flush upserts and deletes to LanceDB.

        The fence is checked again once the upsert returned: a deletion that
        fenced a row after this run filtered it relied on this run to delete
        it (the two share no lock).
        """
        fence = self._deps.fence
        if to_upsert:
            await self.index_repo.upsert(to_upsert)
            late = [r for r in to_upsert if fence.blocks(r)] if fence else []
            if late:
                await self.index_repo.delete(fence.filter_for(late))
        if to_delete_ids:
            await self.index_repo.delete(
                all_of(eq("md_path", md_path), one_of("entry_id", to_delete_ids))
            )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await self.index_repo.delete_by_md_path(md_path)
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
        *,
        entry_ids: Collection[str] | None = None,
    ) -> None:
        """Propagate deprecated_entries from frontmatter to LanceDB, both ways.

        The md file is the source of truth, so the projection sets the
        ``deprecated_by`` column for every mark the file holds **and clears it
        for every entry the file still has without a mark**. The clear is what
        makes an undo of a forget show up: the undo drops the mark, the file's
        bytes change only in the frontmatter, and without it the row would keep
        saying "superseded" forever.

        Only rows whose stored value differs from the file's are written, one
        update per distinct value. Every index update is a commit -- on
        LanceDB each ``update`` call writes a new table version even when it
        matches nothing or changes nothing -- so writing every entry of every
        file on every projection grew rebuilt stores by one version per entry
        per pass (MemLens 2.0: about 173k versions over three tables) and
        queued as many writes on the table's write lock, which then timed out
        as ``VectorStoreBusyError``. The stored values are read once per file,
        after this run's upsert, so a re-embedded row whose mark the upsert
        reset is written again.

        A kind whose record has no ``deprecated_by`` column has nothing to
        propagate and is skipped without touching the index. A row the index
        does not hold (deleted, fenced, not yet indexed) is not written, as
        before.

        Args:
            frontmatter: The file's frontmatter, typed or as a dict.
            owner_id: Whose entries these are.
            app_id: Scope.
            project_id: Scope.
            entry_ids: The entry ids the file currently holds. ``None`` skips
                the clear (a caller that does not know them must not clear
                every row).
        """
        schema = getattr(self.index_repo, "schema", None)
        if schema is not None and "deprecated_by" not in schema.model_fields:
            return
        deprecated = getattr(frontmatter, "deprecated_entries", None)
        if not deprecated and isinstance(frontmatter, dict):
            deprecated = frontmatter.get("deprecated_entries")
        marks = dict(deprecated or {})
        wanted: dict[str, str | None] = {
            entry_id: None for entry_id in (entry_ids or ()) if entry_id not in marks
        }
        wanted.update(marks)
        if not wanted:
            return
        try:
            rows = await self.index_repo.find_by_owner_entries(
                owner_id,
                list(wanted),
                app_id=app_id,
                project_id=project_id,
                skip_vectors=True,
            )
        except Exception:
            logger.warning(
                "cascade_handler_deprecation_read_failed",
                kind=self.kind,
                owner_id=owner_id,
                entries=len(wanted),
                exc_info=True,
            )
            return
        stored = {row.entry_id: row.deprecated_by for row in rows}
        stale: dict[str | None, list[str]] = {}
        for entry_id, value in wanted.items():
            if entry_id in stored and stored[entry_id] != value:
                stale.setdefault(value, []).append(entry_id)
        scope = (app_id, project_id)
        writes = [
            self._mark_deprecated(owner_id, ids, value, scope)
            for value, ids in stale.items()
        ]
        if writes:
            await asyncio.gather(*writes)

    async def _mark_deprecated(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        deprecated_by: str | None,
        scope: tuple[str, str],
    ) -> None:
        """Set (or, with ``None``, clear) ``deprecated_by`` on ``entry_ids``' rows.

        One update for the whole group, scoped to ``(app_id, project_id,
        owner_id)`` plus the entry ids to avoid cross-space collisions.
        """
        app_id, project_id = scope
        predicate = all_of(
            eq("owner_id", owner_id),
            one_of("entry_id", list(entry_ids)),
            eq("app_id", app_id),
            eq("project_id", project_id),
        )
        try:
            await self.index_repo.update(
                {"deprecated_by": deprecated_by},
                where=predicate,
            )
        except Exception:
            logger.warning(
                "failed to mark entry deprecated",
                entry_ids=list(entry_ids),
                deprecated_by=deprecated_by,
                kind=self.kind,
                exc_info=True,
            )

    @abc.abstractmethod
    def _embed_texts(self, entry: ParsedEntry) -> tuple[str, ...]:
        """Subclass: the texts one entry needs embedded, in the order
        :meth:`_build_row` receives their ``vectors`` back."""

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
        vectors: Sequence[list[float] | None] = (),
    ) -> Any:
        """Subclass: build the typed LanceDB row for one parsed entry.

        ``vectors`` aligns with :meth:`_embed_texts`; an empty sequence or
        ``None`` items mean "no vector" (embedding unavailable) and the row
        is written with ``vector=None``.

        ``app_id`` / ``project_id`` carry the path-derived scope; the base
        always supplies them (via :func:`resolve_scope`). They default to
        ``"default"`` so white-box callers exercising only the field mapping
        can omit them.
        """
